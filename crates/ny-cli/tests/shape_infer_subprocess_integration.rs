// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the hidden `ny __shape-infer` subprocess entry.
//!
//! These spawn the REAL `ny` binary (via `CARGO_BIN_EXE_ny`, compiled without
//! `cfg(test)`), so they exercise exactly the child process that
//! `ShapeInferBackend::Subprocess` runs in production:
//! - a valid model round-trips to a versioned shape table on stdout, and the
//!   client backend accepts it end-to-end through `load_onnx_bytes_with_config`;
//! - garbage input exits non-zero (the parent's inference-unavailable signal);
//! - a subprocess that dies without answering degrades the load to the
//!   graceful no-inferred-shapes fallback instead of an error or crash.

use ny_onnx::{onnx_proto, OnnxLoadConfig, ShapeInferBackend, SHAPE_INFER_SUBCOMMAND};
use prost::Message;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn ny_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ny"))
}

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

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        ..Default::default()
    }
}

/// Two-node Add→Relu model whose intermediate tensor "sum" only gets a shape
/// through ORT shape inference (it is not declared in the proto).
fn build_add_relu_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        name: "subprocess_integration_add_relu".to_string(),
        input: vec![
            tensor_value_info("a", &[1, 3]),
            tensor_value_info("b", &[1, 3]),
        ],
        output: vec![tensor_value_info("out", &[1, 3])],
        node: vec![
            node("add", "Add", &["a", "b"], &["sum"]),
            node("relu", "Relu", &["sum"], &["out"]),
        ],
        ..Default::default()
    };
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        graph: Some(graph),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-cli-subprocess-integration".to_string(),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

fn run_shape_infer_child(input: &[u8]) -> std::process::Output {
    let mut child = Command::new(ny_exe())
        .arg(SHAPE_INFER_SUBCOMMAND)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ny __shape-infer");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input)
        .expect("write model bytes");
    // stdin handle dropped by take()+write scope end → EOF for the child.
    child.wait_with_output().expect("collect child output")
}

#[test]
fn real_binary_serves_versioned_shape_table() {
    let output = run_shape_infer_child(&build_add_relu_model_bytes());
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let response: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout is a single JSON document");
    assert_eq!(
        response.get("version").and_then(|v| v.as_u64()),
        Some(1),
        "response must carry the protocol version tag: {response}"
    );
    let shapes = response
        .get("shapes")
        .and_then(|s| s.as_object())
        .expect("response carries a shape table");
    // "sum" is undeclared in the proto — only real ORT inference produces it.
    assert_eq!(
        shapes.get("sum").cloned(),
        Some(serde_json::json!([1, 3])),
        "ORT-inferred intermediate shape must be in the table: {response}"
    );
}

#[test]
fn real_binary_round_trips_through_client_backend() {
    // End-to-end: the ny-onnx loader spawning the real `ny` child must
    // recover the same ORT-inferred intermediate shape as an in-process load.
    let bytes = build_add_relu_model_bytes();
    let config = OnnxLoadConfig::default()
        .with_shape_infer_backend(ShapeInferBackend::Subprocess { exe: ny_exe() });
    let model = ny_onnx::load_onnx_bytes_with_config("add_relu", &bytes, &config)
        .expect("load via subprocess backend");
    assert_eq!(
        model.tensor_shapes().get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice()),
        "subprocess-inferred intermediate shape must reach the loaded model"
    );
}

#[test]
fn real_binary_rejects_garbage_input_with_nonzero_exit() {
    let output = run_shape_infer_child(b"\xff\xfe definitely not an onnx model");
    assert!(
        !output.status.success(),
        "garbage input must fail the child, got stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[cfg(unix)]
#[test]
fn dead_subprocess_degrades_to_graceful_no_shapes_load() {
    // `/usr/bin/false` exits non-zero without answering — the same observable
    // as an ORT SIGABRT under panic=abort. The load must still succeed; only
    // the ORT-inferred intermediate shape is lost.
    let bytes = build_add_relu_model_bytes();
    let config =
        OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::Subprocess {
            exe: PathBuf::from("/usr/bin/false"),
        });
    let model = ny_onnx::load_onnx_bytes_with_config("add_relu", &bytes, &config)
        .expect("dead subprocess must degrade to a no-inferred-shapes load, not an error");
    assert!(
        !model.tensor_shapes().contains_key("sum"),
        "no shapes may be fabricated from a failed subprocess exchange"
    );
    // The declared I/O contract still comes from the proto itself.
    assert_eq!(
        model.tensor_shapes().get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}
