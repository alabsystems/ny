// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #smart-turn-qdq: the dark `qdq_activation_relaxation` loader gate.
//
// DEFAULT OFF must refuse BOTH mechanism halves byte-identically to today:
// (a) a `Cast(to=FLOAT)` fed by a static INT64 structural value (transformer
// attention's `Shape -> Slice -> Cast -> Sqrt` head-dim scaling), and (b) an
// activation `QuantizeLinear -> DequantizeLinear` pair. ON admits exactly
// those two shapes — the Cast only as a deferred-fold obligation (it must be
// erased into a literal by constant folding), the Q->DQ pair only as the
// certified `QdqPerturbationLayer` enclosure — and keeps refusing a Cast
// whose INT64 source is rooted in runtime data.

use ny_onnx::{load_onnx_bytes, load_onnx_bytes_with_config, onnx_proto, OnnxLoadConfig};
use prost::Message;

fn value_info(name: &str, elem_type: i32, shape: &[i64]) -> onnx_proto::ValueInfoProto {
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
                elem_type,
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
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn tensor_i64(name: &str, shape: &[i64], data: &[i64]) -> onnx_proto::TensorProto {
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 7,
        name: name.to_string(),
        int64_data: data.to_vec(),
        ..Default::default()
    }
}

fn tensor_u8(name: &str, shape: &[i64], data: &[i32]) -> onnx_proto::TensorProto {
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 2, // UINT8
        name: name.to_string(),
        int32_data: data.to_vec(),
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
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_type::INT,
        i: Some(value),
        ..Default::default()
    }
}

fn model_bytes(graph: onnx_proto::GraphProto) -> Vec<u8> {
    let model = onnx_proto::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

/// Shape -> Slice -> Cast(FLOAT) -> Sqrt, multiplied into the data path:
/// the attention head-dim scaling pattern, on a statically-shaped input.
fn static_cast_scaling_graph() -> onnx_proto::GraphProto {
    onnx_proto::GraphProto {
        name: "static_cast_scaling".to_string(),
        node: vec![
            node("shape", "Shape", &["x"], &["s"], vec![]),
            node(
                "slice",
                "Slice",
                &["s", "starts", "ends", "axes"],
                &["sl"],
                vec![],
            ),
            node("cast", "Cast", &["sl"], &["c"], vec![int_attr("to", 1)]),
            node("sqrt", "Sqrt", &["c"], &["sq"], vec![]),
            node("mul", "Mul", &["x", "sq"], &["y"], vec![]),
        ],
        input: vec![value_info("x", 1, &[2, 4])],
        output: vec![value_info("y", 1, &[2, 4])],
        initializer: vec![
            tensor_i64("starts", &[1], &[1]),
            tensor_i64("ends", &[1], &[2]),
            tensor_i64("axes", &[1], &[0]),
        ],
        ..Default::default()
    }
}

/// The same Cast-to-FLOAT shape, but rooted in RUNTIME data (ArgMax of the
/// input), which no gate may admit.
/// A runtime-rooted INT64 lane that is NOT a certified f32 readout.
///
/// HISTORY (#qdq-crown-followup): this graph used to be a bare
/// `ArgMax -> Cast(FLOAT) -> Mul`, but #cctsdb-trunc-gated-index later
/// certified exactly that shape as an exact f32 readout (ny models ArgMax
/// natively with f32-valued indices), so the merged tree ADMITS it and the
/// old expectation went stale — this test was red on main for a feature
/// working as designed. The refusal boundary this test pins is one step
/// further out: runtime INT64 values flowing through UNCERTIFIED integer
/// arithmetic (`Add` of two runtime index tensors) before the Cast must
/// still refuse, gate armed or not.
fn runtime_cast_graph() -> onnx_proto::GraphProto {
    onnx_proto::GraphProto {
        name: "runtime_cast".to_string(),
        node: vec![
            node(
                "argmax",
                "ArgMax",
                &["x"],
                &["am"],
                vec![int_attr("axis", 1), int_attr("keepdims", 0)],
            ),
            node("addidx", "Add", &["am", "am"], &["am2"], vec![]),
            node("cast", "Cast", &["am2"], &["c"], vec![int_attr("to", 1)]),
            node("mul", "Mul", &["x", "c"], &["y"], vec![]),
        ],
        input: vec![value_info("x", 1, &[2, 4])],
        output: vec![value_info("y", 1, &[2, 4])],
        ..Default::default()
    }
}

/// One activation QuantizeLinear -> DequantizeLinear pair with shared params.
fn activation_qdq_graph() -> onnx_proto::GraphProto {
    onnx_proto::GraphProto {
        name: "activation_qdq".to_string(),
        node: vec![
            node("relu", "Relu", &["x"], &["a"], vec![]),
            node(
                "q",
                "QuantizeLinear",
                &["a", "scale", "zp"],
                &["qv"],
                vec![],
            ),
            node(
                "dq",
                "DequantizeLinear",
                &["qv", "scale", "zp"],
                &["y"],
                vec![],
            ),
        ],
        input: vec![value_info("x", 1, &[4])],
        output: vec![value_info("y", 1, &[4])],
        initializer: vec![tensor_f32("scale", &[], &[0.1]), tensor_u8("zp", &[], &[3])],
        ..Default::default()
    }
}

fn armed_config() -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_qdq_activation_relaxation(true)
}

#[test]
fn static_cast_to_float_refuses_by_default_exactly_as_today() {
    let bytes = model_bytes(static_cast_scaling_graph());
    let error = load_onnx_bytes("qdq_gate_model", &bytes)
        .expect_err("default path must refuse the FLOAT Cast");
    let message = error.to_string();
    assert!(
        message.contains("reaches data input 0 of 'cast' (Cast)"),
        "default refusal must be the historical structural message, got: {message}"
    );
    // The gate default (unset) must be identical to an explicit `false`.
    let error_explicit_off = load_onnx_bytes_with_config(
        "static_cast_scaling",
        &bytes,
        &OnnxLoadConfig::default().with_qdq_activation_relaxation(false),
    )
    .expect_err("explicit false must refuse identically");
    assert_eq!(message, error_explicit_off.to_string());
}

#[test]
fn static_cast_to_float_loads_with_the_gate_armed() {
    let bytes = model_bytes(static_cast_scaling_graph());
    let model = load_onnx_bytes_with_config("static_cast_scaling", &bytes, &armed_config())
        .expect("armed gate must admit the static Shape->Slice->Cast->Sqrt scaling");
    // The Cast was erased into a literal: nothing integer reaches propagation,
    // and the graph converts end to end.
    model
        .to_graph_network()
        .expect("admitted model must convert to a graph network");
}

#[test]
fn runtime_rooted_cast_still_refuses_with_the_gate_armed() {
    let bytes = model_bytes(runtime_cast_graph());
    let error = load_onnx_bytes_with_config("runtime_cast", &bytes, &armed_config())
        .expect_err("a Cast rooted in runtime data must refuse even with the gate armed");
    let message = error.to_string();
    assert!(
        message.contains("INT64") || message.contains("int64"),
        "the refusal must still name the integer policy, got: {message}"
    );
}

#[test]
fn activation_qdq_pair_refuses_by_default_exactly_as_today() {
    // Default path: the model LOADS (the raw Q/DQ schema is valid), the
    // ny-build converter refuses the activation QuantizeLinear ("rounding is
    // non-linear"), and the graph builder degrades the refused node to the
    // vacuous `OpaqueSkip` `[-inf, +inf]` carrier — no bound, no relaxation.
    // The gate default must preserve exactly that.
    let bytes = model_bytes(activation_qdq_graph());
    let model = load_onnx_bytes("qdq_gate_model", &bytes)
        .expect("the raw activation Q/DQ schema itself is loadable");
    let graph = model
        .to_graph_network()
        .expect("default conversion degrades the refused quantizer to OpaqueSkip");
    let layer_types: Vec<&str> = graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name).map(|node| node.layer().layer_type()))
        .collect();
    assert!(
        layer_types.contains(&"OpaqueSkip"),
        "the refused quantizer must remain the vacuous OpaqueSkip carrier, got {layer_types:?}"
    );
    assert!(
        !layer_types.contains(&"QdqPerturbation"),
        "no relaxation layer may exist on the default path, got {layer_types:?}"
    );
}

#[test]
fn activation_qdq_pair_fuses_to_the_certified_layer_with_the_gate_armed() {
    let bytes = model_bytes(activation_qdq_graph());
    let model = load_onnx_bytes_with_config("activation_qdq", &bytes, &armed_config())
        .expect("armed gate must fuse the activation Q->DQ pair");
    let graph = model
        .to_graph_network()
        .expect("fused model must convert to a graph network");
    let relaxed = graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .is_some_and(|node| node.layer().layer_type() == "QdqPerturbation")
        })
        .count();
    assert_eq!(
        relaxed, 1,
        "exactly one QDQ pair must lower to the certified relaxation layer"
    );
}
