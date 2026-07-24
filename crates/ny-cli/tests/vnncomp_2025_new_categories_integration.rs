// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the 6 new VNN-COMP 2025 categories.
//!
//! These tests verify that ny can load benchmark models, parse VNNLIB
//! properties, and produce valid JSON output for each of the 2025-new categories:
//! malbeware, cersyve, sat_relu, soundnessbench, relusplitter, lsnc_relu.
//!
//! lsnc_relu is already covered in vnncomp_cnn_integration.rs — included here
//! for completeness of the 2025 category coverage matrix.
//!
//! These are smoke tests: they verify model loading, preset application, and
//! JSON output validity. They do NOT assert that instances are solved — that
//! requires feature-specific builds (MIP) and longer timeouts.
//!
//! Part of #3218.

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::workspace_root;
use std::path::PathBuf;
use vnncomp_support::{require_benchmark_file, run_ny};

fn malbeware_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/malbeware")
}

fn cersyve_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/cersyve")
}

fn sat_relu_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/sat_relu")
}

fn soundnessbench_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/soundnessbench")
}

fn relusplitter_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/relusplitter")
}

/// Parse JSON from ny output, accepting any exit code.
///
/// beta-crown returns exit code 0 for Verified/Violated, 2 for Unknown/Timeout.
/// Smoke tests accept any exit code as long as JSON output is valid.
fn parse_json_any_exit(output: &std::process::Output, label: &str) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON output for {label}: {e}\nstdout: {stdout}\nstderr: {stderr}")
    })
}

// ---------------------------------------------------------------------------
// malbeware — Malware image classification (MalImg dataset)
// ---------------------------------------------------------------------------

/// malbeware linear-25 model loads and produces valid JSON with preset.
///
/// Verifies the simplest malbeware model (linear Conv→ReLU→Flatten→Gemm, 4096→25)
/// loads correctly with the preset. The linear-25 eps-1 instance verifies via
/// alpha-CROWN alone (no MIP needed).
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_malbeware_preset_loads_linear_model() {
    let category_dir = malbeware_dir();
    let model_path = category_dir.join("onnx/malware_malimg_family_scaled_linear-25.onnx");
    let property_path =
        category_dir.join("vnnlib/malbeware_family-Obfuscator.AD_label-17_eps-1_idx-89.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/malbeware.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    // No --complete-verifier mip: MIP requires the `mip` feature flag.
    // The linear-25 eps-1 instance verifies via alpha-CROWN alone.
    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset_path.to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "10",
        "--max-domains",
        "10",
        "--pgd-attack",
        "--json",
    ]);

    let json = parse_json_any_exit(&output, "malbeware linear-25");
    let status = json["status"]
        .as_str()
        .expect("expected status field in malbeware JSON output");
    assert!(
        [
            "verified", "Verified", "violated", "Violated", "timeout", "Timeout", "unknown",
            "Unknown"
        ]
        .contains(&status),
        "unexpected status for malbeware: {status}"
    );
}

// ---------------------------------------------------------------------------
// cersyve — Controller safety verification (lane keep, pendulum, point mass)
// ---------------------------------------------------------------------------

/// cersyve DAG model loads with preset input splitting.
///
/// Verifies the cersyve lane_keep_pretrain_con model (small DAG, 28 layers,
/// 1288 params) loads with the input-split preset and produces valid JSON.
/// This is a SAT (pretrain) instance — PGD should find a counterexample quickly.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_cersyve_preset_loads_dag_model() {
    let category_dir = cersyve_dir();
    let model_path = category_dir.join("onnx/lane_keep_pretrain_con.onnx");
    let property_path = category_dir.join("vnnlib/prop_lane_keep.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/cersyve.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset_path.to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "10",
        "--max-domains",
        "10",
        "--pgd-attack",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Model is a DAG"),
        "preset-driven cersyve invocation should not trip the DAG branching guard.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_any_exit(&output, "cersyve");
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

// ---------------------------------------------------------------------------
// sat_relu — SAT-encoded ReLU networks (NeuroCodeBench 2.0)
// ---------------------------------------------------------------------------

/// sat_relu model loads and BaB produces valid JSON output.
///
/// sat_relu networks are SAT-encoded: CROWN relaxation is fundamentally too
/// loose, so the reference uses MIP exclusively. This test verifies model
/// loading, VNNLIB parsing, and BaB produce valid JSON. MIP routing requires
/// the `mip` feature flag and is tested separately in benchmark runs.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_sat_relu_preset_loads_model() {
    let category_dir = sat_relu_dir();
    let model_path = category_dir.join("onnx/sat_v30_c38.onnx");
    let property_path = category_dir.join("vnnlib/sat_v30_c38.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/sat_relu.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    // No MIP flags — MIP requires feature flag. BaB alone will timeout/unknown
    // for SAT-encoded networks, but the test verifies model loading + JSON.
    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset_path.to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "3",
        "--max-domains",
        "5",
        "--json",
    ]);

    let json = parse_json_any_exit(&output, "sat_relu");
    let status = json["status"]
        .as_str()
        .expect("expected status field in sat_relu JSON output");
    assert!(
        [
            "verified", "Verified", "violated", "Violated", "timeout", "Timeout", "unknown",
            "Unknown"
        ]
        .contains(&status),
        "unexpected status for sat_relu: {status}"
    );
}

// ---------------------------------------------------------------------------
// soundnessbench — Large feedforward model soundness benchmark
// ---------------------------------------------------------------------------

/// soundnessbench model loads and IBP produces valid JSON output.
///
/// The soundnessbench model is large (1.74M params, 128→384, 16 layers). CROWN
/// setup for 384 output dimensions takes >300s on CPU, so we test with IBP only
/// (`ny verify --method ibp`) which is instant. Full verification requires
/// GPU CROWN and is tested via benchmark scripts.
#[ntest::timeout(120000)]
#[test]
fn test_soundnessbench_model_loads_with_ibp() {
    let category_dir = soundnessbench_dir();
    let model_path = category_dir.join("onnx/model.onnx");
    let property_path = category_dir.join("vnnlib/model_0.vnnlib");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    // Use `ny verify --method ibp` instead of beta-crown: CROWN setup for
    // 384 outputs takes >300s on CPU (1.74M param model). IBP verifies model
    // loading and VNNLIB parsing in seconds.
    let output = run_ny(&[
        "verify",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--method",
        "ibp",
        "--timeout",
        "30",
        "--json",
    ]);

    let json = parse_json_any_exit(&output, "soundnessbench");
    assert!(
        json.get("status").is_some(),
        "expected verify JSON output to include a status field: {json}"
    );
}

// ---------------------------------------------------------------------------
// relusplitter — ReLU splitting benchmark (MNIST FC + CIFAR CNN)
// ---------------------------------------------------------------------------

/// relusplitter MNIST model loads and preset kFSB branching produces JSON output.
///
/// Verifies the simplest relusplitter model (MNIST FC 256x4) loads correctly
/// with the kFSB preset. The disjunctive property may cause timeout/unknown
/// within the short test budget — that is expected.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_relusplitter_preset_loads_mnist_model() {
    let category_dir = relusplitter_dir();
    let model_path = category_dir.join("onnx/mnist_fc_vnncomp2022_mnist-net_256x4.onnx");
    let property_path = category_dir.join("vnnlib/mnist_fc_vnncomp2022_prop_5_0.05.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/relusplitter.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset_path.to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "5",
        "--max-domains",
        "10",
        "--pgd-attack",
        "--json",
    ]);

    let json = parse_json_any_exit(&output, "relusplitter");
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}
