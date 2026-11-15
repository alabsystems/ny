// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Black-box tests for the public process-exit and JSON-error contract.

use std::path::PathBuf;
use std::process::{Command, Output};

const OPERATIONAL_ERROR: i32 = 4;

fn run_ny(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ny"))
        .args(args)
        .output()
        .expect("execute ny CLI")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/models")
        .join(name)
}

fn assert_exit(output: &Output, expected: i32) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "unexpected exit code\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn parse_stdout_json(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout is not JSON: {error}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

#[test]
fn clap_usage_errors_use_operational_exit_code() {
    let output = run_ny(&["verify", "--definitely-not-a-real-option"]);
    assert_exit(&output, OPERATIONAL_ERROR);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unexpected argument"),
        "clap diagnostic should remain on stderr"
    );
}

#[test]
fn clap_help_remains_successful() {
    let output = run_ny(&["--help"]);
    assert_exit(&output, 0);
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("Usage:"),
        "help should be printed to stdout"
    );
}

#[test]
fn verify_rejects_explicit_backend_with_legacy_gpu_flag() {
    let model = fixture("single_linear.onnx");
    let output = run_ny(&[
        "verify",
        model.to_str().expect("UTF-8 fixture path"),
        "--backend",
        "cpu",
        "--gpu",
    ]);
    assert_exit(&output, OPERATIONAL_ERROR);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("cannot be used with"),
        "clap should explain the conflicting backend flags"
    );
}

#[test]
fn verify_pre_dispatch_json_error_is_stable_and_operational() {
    let output = run_ny(&["verify", "--json"]);
    assert_exit(&output, OPERATIONAL_ERROR);
    let json = parse_stdout_json(&output);
    assert_eq!(json["error"], "verify_failed");
    assert!(
        json["message"]
            .as_str()
            .is_some_and(|message| message.contains("MODEL is required")),
        "unexpected JSON error: {json}"
    );
}

#[test]
fn verify_allow_unknown_is_an_explicit_unknown_only_override() {
    let model = fixture("crossing_relu.nnet");
    let property = fixture("crossing_relu_unsafe.vnnlib");
    let output = run_ny(&[
        "verify",
        model.to_str().expect("UTF-8 fixture path"),
        "--property",
        property.to_str().expect("UTF-8 fixture path"),
        "--method",
        "ibp",
        "--backend",
        "cpu",
        "--json",
        "--allow-unknown",
    ]);
    assert_exit(&output, 0);
    let json = parse_stdout_json(&output);
    assert_eq!(json["status"], "unknown");
    assert_eq!(json["property_status"], "unknown");
}

#[test]
fn verify_allow_unknown_does_not_mask_timeout() {
    let model = fixture("crossing_relu.nnet");
    let property = fixture("crossing_relu_safe.vnnlib");
    let output = run_ny(&[
        "verify",
        model.to_str().expect("UTF-8 fixture path"),
        "--property",
        property.to_str().expect("UTF-8 fixture path"),
        "--method",
        "beta",
        "--timeout",
        "0",
        "--backend",
        "cpu",
        "--json",
        "--allow-unknown",
    ]);
    assert_exit(&output, 3);
    let json = parse_stdout_json(&output);
    assert_eq!(json["status"], "timeout");
}

#[test]
fn beta_crown_json_load_error_is_not_an_unknown_verdict() {
    let temp = tempfile::tempdir().expect("temporary missing-model directory");
    let missing = temp.path().join("missing.onnx");
    let output = run_ny(&[
        "beta-crown",
        missing.to_str().expect("UTF-8 temp path"),
        "--backend",
        "cpu",
        "--no-certificate",
        "--json",
    ]);
    assert_exit(&output, OPERATIONAL_ERROR);
    let json = parse_stdout_json(&output);
    assert_eq!(json["error"], "beta_crown_failed");
    assert!(
        json.get("status").is_none(),
        "error must not mimic a verdict"
    );
    assert!(json["message"].is_string(), "unexpected JSON error: {json}");
}

#[test]
fn beta_crown_text_load_error_is_operational_not_falsified() {
    let temp = tempfile::tempdir().expect("temporary missing-model directory");
    let missing = temp.path().join("missing.onnx");
    let output = run_ny(&[
        "beta-crown",
        missing.to_str().expect("UTF-8 temp path"),
        "--backend",
        "cpu",
        "--no-certificate",
    ]);
    assert_exit(&output, OPERATIONAL_ERROR);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Error:"),
        "text operational error should be rendered on stderr"
    );
}

#[test]
fn beta_crown_readme_fixtures_match_verdict_exit_codes() {
    let model = fixture("crossing_relu.nnet");
    for (property_name, expected_status, expected_exit) in [
        ("crossing_relu_safe.vnnlib", "verified", 0),
        ("crossing_relu_unsafe.vnnlib", "violated", 1),
    ] {
        let property = fixture(property_name);
        let output = run_ny(&[
            "beta-crown",
            model.to_str().expect("UTF-8 fixture path"),
            "--property",
            property.to_str().expect("UTF-8 fixture path"),
            "--backend",
            "cpu",
            "--no-certificate",
            "--json",
        ]);
        assert_exit(&output, expected_exit);
        let json = parse_stdout_json(&output);
        assert_eq!(
            json["status"], expected_status,
            "README fixture status and exit code must describe the same verdict"
        );
    }
}

#[test]
fn beta_crown_readme_safe_fixture_writes_default_certificate() {
    let temp = tempfile::tempdir().expect("temporary README smoke directory");
    let model = temp.path().join("crossing_relu.nnet");
    let property = temp.path().join("crossing_relu_safe.vnnlib");
    std::fs::copy(fixture("crossing_relu.nnet"), &model).expect("copy README model fixture");
    std::fs::copy(fixture("crossing_relu_safe.vnnlib"), &property)
        .expect("copy README property fixture");

    let output = run_ny(&[
        "beta-crown",
        model.to_str().expect("UTF-8 temp path"),
        "--property",
        property.to_str().expect("UTF-8 temp path"),
        "--backend",
        "cpu",
        "--json",
    ]);
    assert_exit(&output, 0);
    let json = parse_stdout_json(&output);
    assert_eq!(json["status"], "verified");

    let certificate = model.with_extension("cert.json");
    let metadata = std::fs::metadata(&certificate).unwrap_or_else(|error| {
        panic!(
            "default certificate {} was not created: {error}",
            certificate.display()
        )
    });
    assert!(
        metadata.len() > 0,
        "default certificate {} is empty",
        certificate.display()
    );
    let certificate_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&certificate).expect("read generated default certificate"),
    )
    .expect("generated default certificate is JSON");
    assert_eq!(certificate_json["format"], "ny-cert/crown-deep/v1");
}

#[test]
fn onnx_verify_beta_rejects_alpha_compatibility_fallback() {
    let model = fixture("single_linear.onnx");
    let output = run_ny(&[
        "verify",
        model.to_str().expect("UTF-8 fixture path"),
        "--method",
        "beta",
        "--backend",
        "cpu",
        "--json",
    ]);
    assert_exit(&output, OPERATIONAL_ERROR);
    let json = parse_stdout_json(&output);
    assert_eq!(json["error"], "verify_failed");
    assert!(
        json["message"]
            .as_str()
            .is_some_and(|message| message.contains("use `ny beta-crown")),
        "rejection should direct users to the complete command: {json}"
    );
}

#[test]
fn ground_truth_load_error_is_operational_not_falsified() {
    let temp = tempfile::tempdir().expect("temporary missing-input directory");
    let model = temp.path().join("missing.onnx");
    let spec = temp.path().join("missing.gt.json");
    let output = run_ny(&[
        "gt",
        "verify",
        model.to_str().expect("UTF-8 temp path"),
        spec.to_str().expect("UTF-8 temp path"),
    ]);
    assert_exit(&output, OPERATIONAL_ERROR);
}

// ---------------------------------------------------------------------------
// `ny proof-check` (ENGINE_PROGRAM §1.7 B2.1)
// ---------------------------------------------------------------------------

fn proof_check_mlp_onnx() -> Vec<u8> {
    use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
    use ny_onnx::onnx_proto::{
        AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
        TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use prost::Message;
    let vi = |name: &str, dims: &[i64]| ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(dimension::Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    };
    let t = |name: &str, dims: &[i64], v: &[f32]| TensorProto {
        name: name.to_owned(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: v.to_vec(),
        ..Default::default()
    };
    let gemm = |inputs: [&str; 3], out: &str| NodeProto {
        op_type: "Gemm".to_owned(),
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: vec![out.to_owned()],
        attribute: vec![AttributeProto {
            name: "transB".to_owned(),
            i: Some(1),
            r#type: 2,
            ..Default::default()
        }],
        ..Default::default()
    };
    // z = [x0+x1, x0-x1, x0+x1+5]; y0 = relu(z0) + relu(z1) + 2^-20, y1 = 0.
    let model = ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: vec![
                gemm(["x", "W1", "B1"], "z"),
                NodeProto {
                    op_type: "Relu".to_owned(),
                    input: vec!["z".to_owned()],
                    output: vec!["h".to_owned()],
                    ..Default::default()
                },
                gemm(["h", "W2", "B2"], "y"),
            ],
            initializer: vec![
                t("W1", &[3, 2], &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
                t("B1", &[3], &[0.0, 0.0, 5.0]),
                t("W2", &[2, 3], &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                t("B2", &[2], &[1.0 / 1_048_576.0, 0.0]),
            ],
            input: vec![vi("x", &[1, 2])],
            output: vec![vi("y", &[1, 2])],
            ..Default::default()
        }),
        ..Default::default()
    };
    model.encode_to_vec()
}

fn proof_check_bundle(
    onnx: &std::path::Path,
    vnnlib: &std::path::Path,
    leaves: &[(Vec<[u64; 3]>, Vec<f64>)],
) -> Vec<u8> {
    proof_check_bundle_closed(onnx, vnnlib, leaves, "bound")
}

/// [`proof_check_bundle`] with every node closed as `close` (`bound` or `opaque`).
fn proof_check_bundle_closed(
    onnx: &std::path::Path,
    vnnlib: &std::path::Path,
    leaves: &[(Vec<[u64; 3]>, Vec<f64>)],
    close: &str,
) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let digest = |p: &std::path::Path| {
        let b = std::fs::read(p).expect("fixture");
        (format!("{:x}", Sha256::digest(&b)), b.len())
    };
    let (osha, olen) = digest(onnx);
    let (vsha, vlen) = digest(vnnlib);
    let mut lines = vec![
        serde_json::json!({
            "kind": "header", "schema": "ny-proof-bundle/v2", "build": "cli-contract",
            "profile": "proof", "profile_source": "cli", "lane": "margin_row",
            "site_space": "onnx-relu-input/v1", "input_space": "onnx-input0-flat/v1",
            "onnx": {"path": onnx.display().to_string(), "sha256": osha, "bytes": olen},
            "vnnlib": {"path": vnnlib.display().to_string(), "sha256": vsha, "bytes": vlen},
        }),
        serde_json::json!({"kind": "sites", "tensors": [{"id": 1, "name": "z", "size": 3, "op": 1}]}),
        serde_json::json!({"kind": "clauses", "region": 0, "list": [{"id": 1, "atoms": ["Y0:1,Y1:-1<=0"]}]}),
    ];
    for (k, (lits, beta)) in leaves.iter().enumerate() {
        lines.push(serde_json::json!({
            "kind": "node", "id": k, "region": 0, "lits": lits, "close": close, "src": "tree",
            "dis": [1], "hint": {"beta": beta, "lift": false, "margin": 1e-6},
        }));
    }
    lines.push(serde_json::json!({
        "kind": "verdict", "source": "authorized", "required": [1], "min_margin": 1e-6,
        "nodes": leaves.len(),
    }));
    lines.push(serde_json::json!({"kind": "published", "result": "unsat"}));
    let mut body = String::new();
    for l in &lines {
        body.push_str(&l.to_string());
        body.push('\n');
    }
    let end = serde_json::json!({
        "kind": "end",
        "lines": lines.len() + 1,
        "sha256": format!("{:x}", Sha256::digest(body.as_bytes())),
    });
    body.push_str(&end.to_string());
    body.push('\n');
    body.into_bytes()
}

/// ENGINE_PROGRAM §1.7 B2.1: `ny proof-check` exits with the documented
/// code — 0 verified, 10 unconfirmed, 20 invalid, 30 malformed, 4 for a usage
/// error or an unreadable input — and its `PROOF_CHECK_DONE` line carries the
/// REAL code first (`rc=<code> `), on stderr and as the last stdout line.
#[test]
fn cli_proof_check_exit_codes_are_the_documented_contract() {
    let dir = tempfile::tempdir().expect("tempdir");
    let onnx = dir.path().join("mlp.onnx");
    let vnnlib = dir.path().join("prop.vnnlib");
    std::fs::write(&onnx, proof_check_mlp_onnx()).expect("onnx");
    std::fs::write(
        &vnnlib,
        "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
         (declare-const Y_1 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n\
         (assert (>= X_1 -1))\n(assert (<= X_1 1))\n(assert (or (and (>= Y_1 Y_0))))\n",
    )
    .expect("vnnlib");
    let leaves = vec![
        (vec![[1, 0, 1], [1, 1, 1]], vec![1.0, 1.0]),
        (vec![[1, 0, 1], [1, 1, 0]], vec![1.0, 0.0]),
        (vec![[1, 0, 0], [1, 1, 1]], vec![0.0, 1.0]),
        (vec![[1, 0, 0], [1, 1, 0]], vec![0.0, 0.0]),
    ];
    let check = |name: &str, bytes: &[u8], expected: i32| {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).expect("bundle");
        let output = run_ny(&[
            "proof-check",
            path.to_str().expect("UTF-8"),
            "--threads",
            "2",
        ]);
        assert_exit(&output, expected);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let done = format!("PROOF_CHECK_DONE rc={expected} ");
        assert!(
            stderr.contains(&done),
            "{name}: stderr lacks `{done}`:\n{stderr}"
        );
        assert!(
            stderr.contains("[proof-check] verdict="),
            "{name}:\n{stderr}"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.lines().last().is_some_and(|l| l.starts_with(&done)),
            "{name}: the last stdout line is the DONE line:\n{stdout}"
        );
    };
    check(
        "ok.proof.jsonl",
        &proof_check_bundle(&onnx, &vnnlib, &leaves),
        0,
    );
    // Unconfirmed (10) needs a node the checker cannot discharge. A root-only
    // `bound` node no longer qualifies: since B2.2 the clip rung confirms it on
    // this true property. An `opaque` route claim is never confirmed.
    check(
        "opaque.proof.jsonl",
        &proof_check_bundle_closed(&onnx, &vnnlib, &[(Vec::new(), Vec::new())], "opaque"),
        10,
    );
    check(
        "gap.proof.jsonl",
        &proof_check_bundle(&onnx, &vnnlib, &leaves[..3]),
        20,
    );
    check("garbage.proof.jsonl", b"not a bundle\n", 30);
    // Operational: an unreadable bundle and a usage error.
    let missing = dir.path().join("missing.proof.jsonl");
    let output = run_ny(&["proof-check", missing.to_str().expect("UTF-8")]);
    assert_exit(&output, OPERATIONAL_ERROR);
    let output = run_ny(&["proof-check", "x.proof.jsonl", "--refine", "maybe"]);
    assert_exit(&output, OPERATIONAL_ERROR);
}
