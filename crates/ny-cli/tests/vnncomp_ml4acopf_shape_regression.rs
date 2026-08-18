// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "external-vnncomp")]

//! Regression tests for the real `ml4acopf_2024` graph BaB path.
//!
//! These cover the historical `#3602` shape mismatch on disjunctive DAG
//! properties and pin one concrete solved row from the same benchmark family.

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::workspace_root;
use std::path::PathBuf;
use std::process::Output;
use vnncomp_support::{parse_json_output, require_benchmark_file, run_ny};

const VALID_EXIT_CODES: [i32; 4] = [0, 1, 2, 3];

fn ml4acopf_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/ml4acopf_2024")
}

fn run_ml4acopf_input_split_probe(model_name: &str, property_name: &str) -> Output {
    let category_dir = ml4acopf_dir();
    let model_path = category_dir.join(format!("onnx/{model_name}"));
    let property_path = category_dir.join(format!("vnnlib/{property_name}"));
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--complete-verifier",
        "bab",
        "--branching",
        "input",
        "--timeout",
        "10",
        "--no-alpha",
        "--json",
    ])
}

fn assert_valid_verifier_exit(output: &Output, command_label: &str) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(
        VALID_EXIT_CODES.contains(&exit_code),
        "{command_label} exited with unexpected code {exit_code}.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

/// Regression test for #3602: the exact 14_ieee prop9 disjunctive DAG case
/// must not regress to `Shape mismatch: expected [1], got [2]`.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_ml4acopf_disjunctive_property_avoids_shape_mismatch_3602() {
    let output = run_ml4acopf_input_split_probe("14_ieee_ml4acopf.onnx", "14_ieee_prop9.vnnlib");
    assert_valid_verifier_exit(&output, "beta-crown ml4acopf prop9");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "ml4acopf disjunctive DAG probe should not regress to the #3602 shape mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown ml4acopf prop9");
    assert!(
        json["status"].is_string(),
        "beta-crown ml4acopf prop9 must emit a JSON status: {json}"
    );
}

/// Acceptance guard for #3602: at least one real ml4acopf graph BaB invocation
/// should now resolve to a concrete verification verdict instead of crashing.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_ml4acopf_returns_verified_verdict() {
    let output = run_ml4acopf_input_split_probe("14_ieee_ml4acopf.onnx", "14_ieee_prop3.vnnlib");
    assert_valid_verifier_exit(&output, "beta-crown ml4acopf prop3");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "ml4acopf solved probe should not surface a shape mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown ml4acopf prop3");
    assert_eq!(
        json["status"].as_str(),
        Some("verified"),
        "14_ieee prop3 should remain a quick verified row on current HEAD: {json}"
    );
}
