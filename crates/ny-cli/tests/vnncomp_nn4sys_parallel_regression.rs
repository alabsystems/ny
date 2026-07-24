// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for the real nn4sys parallel pensieve lane.
//!
//! This pins the `pensieve_big_parallel` path that regressed from running BaB
//! to an immediate `error` during the March 2026 #4354 investigation.

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::workspace_root;
use std::path::PathBuf;
use std::process::Output;
use vnncomp_support::{parse_json_output, require_benchmark_file, run_ny};

const VALID_EXIT_CODES: [i32; 4] = [0, 1, 2, 3];

fn nn4sys_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/nn4sys")
}

fn run_nn4sys_parallel_probe() -> Output {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/pensieve_big_parallel.onnx");
    let property_path = category_dir.join("vnnlib/pensieve_parallel_1.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/nn4sys.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset_path.to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "1",
        "--max-domains",
        "1",
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

#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_big_parallel_probe_reaches_bab_4354() {
    let output = run_nn4sys_parallel_probe();
    assert_valid_verifier_exit(&output, "beta-crown nn4sys big parallel probe");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "nn4sys big parallel probe should not regress to a shape-mismatch failure.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown nn4sys big parallel probe");
    assert_ne!(
        json["status"].as_str(),
        Some("error"),
        "nn4sys big parallel probe should reach BaB instead of failing immediately: {json}"
    );
    assert!(
        json["domains_explored"].as_u64().unwrap_or_default() >= 1,
        "nn4sys big parallel probe should explore at least one domain before stopping: {json}"
    );
}
