// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "external-vnncomp")]

//! Integration tests covering real `ml4acopf_2024` benchmark assets.

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::workspace_root;
use std::path::PathBuf;
use std::process::Output;
use vnncomp_support::{parse_json_output, require_benchmark_file, run_ny};

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
        "--branching",
        "input",
        "--timeout",
        "10",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ])
}

/// Real ml4acopf DAG benchmark must degrade shape-mismatching graph CROWN paths
/// to IBP instead of aborting the CLI.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_ml4acopf_input_split_falls_back_on_shape_mismatch() {
    let output = run_ml4acopf_input_split_probe("14_ieee_ml4acopf.onnx", "14_ieee_prop2.vnnlib");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "ml4acopf input split should fall back to IBP instead of surfacing a shape mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown ml4acopf input split");
    assert!(
        matches!(
            json["status"].as_str(),
            Some("unknown" | "verified" | "violated")
        ),
        "ml4acopf input split should return a real verifier verdict: {json}"
    );
}

/// Regression test for #3602: the exact 14_ieee prop9 disjunctive DAG case must
/// not re-surface the historical `expected [1], got [2]` shape mismatch.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_ml4acopf_disjunctive_property_avoids_shape_mismatch_3602() {
    let output = run_ml4acopf_input_split_probe("14_ieee_ml4acopf.onnx", "14_ieee_prop9.vnnlib");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "ml4acopf disjunctive DAG probe should not regress to the #3602 shape mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown ml4acopf disjunctive DAG probe");
    let status = json["status"].as_str();
    assert!(
        matches!(status, Some("unknown" | "verified" | "violated")),
        "disjunctive ml4acopf probe should return a real verifier verdict: {json}"
    );
}
