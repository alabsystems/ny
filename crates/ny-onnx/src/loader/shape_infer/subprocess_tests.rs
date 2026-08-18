// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the subprocess shape-inference protocol: server/parse round-trip,
//! every client failure mode mapping to `Err` (never a panic), and the
//! loader-level guarantee that a dead subprocess degrades to the graceful
//! no-inferred-shapes fallback instead of failing the model load.

use super::subprocess::{
    infer_tensor_shapes_via_subprocess, parse_shape_infer_response, serve_shape_infer_request,
};
use crate::loader::{load_onnx_bytes_with_config, OnnxLoadConfig, ShapeInferBackend};
use crate::onnx_proto;
use prost::Message;
// Used only by the `#[cfg(unix)]` tests below (they build executable shell
// scripts, which needs Unix mode bits); gated to match its uses.
#[cfg(unix)]
use std::path::PathBuf;

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

fn model_bytes_from_graph(graph: onnx_proto::GraphProto) -> Vec<u8> {
    // ir_version / opset literals match the shape_infer defaults; spelled out
    // here so these tests compile with and without the `ort` feature (the
    // normalization helpers are `ort`-gated).
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        graph: Some(graph),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: super::DEFAULT_OPSET_VERSION,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-subprocess-test".to_string(),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

fn build_add_relu_model_bytes() -> Vec<u8> {
    model_bytes_from_graph(onnx_proto::GraphProto {
        name: "subprocess_add_relu".to_string(),
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
    })
}

fn build_relu_model_bytes() -> Vec<u8> {
    model_bytes_from_graph(onnx_proto::GraphProto {
        name: "subprocess_relu".to_string(),
        input: vec![tensor_value_info("x", &[1, 3])],
        output: vec![tensor_value_info("out", &[1, 3])],
        node: vec![node("relu", "Relu", &["x"], &["out"])],
        ..Default::default()
    })
}

#[cfg(unix)]
fn write_script(dir: &tempfile::TempDir, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.path().join(name);
    std::fs::write(&path, body).expect("write script");
    let mut perms = std::fs::metadata(&path).expect("stat script").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).expect("chmod script");
    path
}

// ---------------------------------------------------------------------------
// Server + response codec round-trip (no process spawn).
// ---------------------------------------------------------------------------

#[test]
fn serve_round_trips_shapes_through_versioned_payload() {
    let bytes = build_add_relu_model_bytes();
    let mut payload = Vec::new();
    serve_shape_infer_request(&mut bytes.as_slice(), &mut payload).expect("serve");

    let shapes = parse_shape_infer_response(&payload).expect("parse served payload");
    // The wire round-trip must reproduce exactly what the in-process path
    // computes (with `ort` disabled both are the empty table, so this equality
    // is feature-independent).
    let direct = super::infer_tensor_shapes_from_ort(&bytes).expect("direct in-process inference");
    assert_eq!(shapes, direct);

    #[cfg(feature = "ort")]
    {
        assert_eq!(
            shapes.get("sum").map(Vec::as_slice),
            Some([1, 3].as_slice()),
            "intermediate tensor shape must survive the wire round-trip"
        );
        assert_eq!(
            shapes.get("out").map(Vec::as_slice),
            Some([1, 3].as_slice())
        );
    }
}

// Without `ort` the server's inference stub ignores the payload (empty table),
// so the rejection is only observable when real inference runs.
#[cfg(feature = "ort")]
#[test]
fn serve_rejects_non_model_input() {
    // Garbage stdin must yield Err on the server side, so the CLI entry exits
    // non-zero and the parent observes a protocol failure — never fabricated
    // shapes.
    let mut payload = Vec::new();
    let mut garbage: &[u8] = b"\xff\xfe not an onnx model";
    assert!(serve_shape_infer_request(&mut garbage, &mut payload).is_err());
}

#[test]
fn parse_rejects_version_mismatch() {
    let payload = br#"{"version": 999, "shapes": {}}"#;
    let err = parse_shape_infer_response(payload).expect_err("version 999 must be rejected");
    assert!(
        err.to_string().contains("version"),
        "error must name the version mismatch: {err}"
    );
}

#[test]
fn parse_rejects_garbage_output() {
    // A binary that does not serve the protocol (e.g. a test harness handed
    // `__shape-infer` as a filter) prints something like this to stdout.
    let err = parse_shape_infer_response(b"running 0 tests\n").expect_err("garbage rejected");
    assert!(
        err.to_string().contains("unparseable"),
        "error must flag unparseable output: {err}"
    );
}

// ---------------------------------------------------------------------------
// Client failure modes: every one is Err, never a panic or fake success.
// ---------------------------------------------------------------------------

#[test]
fn subprocess_missing_exe_is_error_not_panic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("no-such-binary");
    let err = infer_tensor_shapes_via_subprocess(&missing, &build_relu_model_bytes())
        .expect_err("missing exe must fail");
    assert!(
        err.to_string().contains("spawn"),
        "error must report the spawn failure: {err}"
    );
}

#[cfg(unix)]
#[test]
fn subprocess_dying_child_is_error() {
    // `/usr/bin/false` exits non-zero without reading stdin or writing stdout —
    // the same observable as an ORT SIGABRT under panic=abort.
    let err = infer_tensor_shapes_via_subprocess(
        std::path::Path::new("/usr/bin/false"),
        &build_relu_model_bytes(),
    )
    .expect_err("dead child must fail");
    assert!(
        err.to_string().contains("exited with"),
        "error must report the exit status: {err}"
    );
}

#[cfg(unix)]
#[test]
fn subprocess_fake_server_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let exe = write_script(
        &dir,
        "fake-server.sh",
        "#!/bin/sh\ncat >/dev/null\nprintf '%s' '{\"version\":1,\"shapes\":{\"x\":[1,3]}}'\n",
    );
    let shapes = infer_tensor_shapes_via_subprocess(&exe, &build_relu_model_bytes())
        .expect("valid protocol answer accepted");
    assert_eq!(shapes.get("x").map(Vec::as_slice), Some([1, 3].as_slice()));
    assert_eq!(shapes.len(), 1);
}

#[cfg(unix)]
#[test]
fn subprocess_version_mismatch_is_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let exe = write_script(
        &dir,
        "wrong-version.sh",
        "#!/bin/sh\ncat >/dev/null\nprintf '%s' '{\"version\":2,\"shapes\":{}}'\n",
    );
    let err = infer_tensor_shapes_via_subprocess(&exe, &build_relu_model_bytes())
        .expect_err("wrong protocol version must be rejected");
    assert!(err.to_string().contains("version"), "{err}");
}

#[cfg(unix)]
#[test]
fn subprocess_zero_exit_with_garbage_stdout_is_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let exe = write_script(
        &dir,
        "garbage.sh",
        "#!/bin/sh\ncat >/dev/null\necho 'running 0 tests'\n",
    );
    let err = infer_tensor_shapes_via_subprocess(&exe, &build_relu_model_bytes())
        .expect_err("garbage stdout must be rejected even on exit 0");
    assert!(err.to_string().contains("unparseable"), "{err}");
}

// ---------------------------------------------------------------------------
// Loader-level fail-closed guarantee: a dead subprocess degrades exactly like
// today's inference-unavailable path (model loads with no inferred shapes),
// never an error and never a crash.
// ---------------------------------------------------------------------------

#[test]
fn loader_degrades_gracefully_when_subprocess_exe_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config =
        OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::Subprocess {
            exe: dir.path().join("no-such-binary"),
        });
    let model = load_onnx_bytes_with_config("relu", &build_relu_model_bytes(), &config)
        .expect("load must degrade to no inferred shapes, not fail");
    assert_eq!(model.network.layers.len(), 1);
}

#[cfg(unix)]
#[test]
fn loader_degrades_gracefully_when_subprocess_dies() {
    let config =
        OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::Subprocess {
            exe: PathBuf::from("/usr/bin/false"),
        });
    let model = load_onnx_bytes_with_config("relu", &build_relu_model_bytes(), &config)
        .expect("load must degrade to no inferred shapes, not fail");
    assert_eq!(model.network.layers.len(), 1);
}
