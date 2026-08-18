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
