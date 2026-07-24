// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests covering real VNN-COMP CNN benchmark assets.
//!
//! These tests use CIFAR-10 ResNet benchmark files from `benchmarks/vnncomp2021`
//! to ensure ny can load real CNN benchmark data through CLI entry points.
//!
//! Part of #2665.

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::workspace_root;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use vnncomp_support::{parse_json_output, require_benchmark_file, run_ny};

const VALID_EXIT_CODES: [i32; 4] = [0, 1, 2, 3];

fn cifar10_resnet_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2021/benchmarks/cifar10_resnet")
}

fn lsnc_relu_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/lsnc_relu")
}

fn linearizenn_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/linearizenn_2024")
}

fn nn4sys_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/nn4sys")
}

fn tinyimagenet_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/tinyimagenet_2024")
}

fn cgan_dir() -> PathBuf {
    workspace_root().join("benchmarks/vnncomp2025/benchmarks/cgan_2023")
}

/// Regression test for the `ny vnncomp` COMPETITION ENTRY POINT on a DAG conv model.
///
/// The `vnncomp` subcommand invokes `handle_beta_crown_command` with `branching:
/// None` (meaning "auto"), whereas the `beta-crown` clap subcommand defaults
/// `--branching` to the string `"auto"`. Auto-branching used to be gated on
/// `cli_branching_auto = (branching == Some("auto"))`, so the `None` the `vnncomp`
/// path passes silently DISABLED model-class auto-branching — every DAG model
/// (ResNet/ViT: tinyimagenet, cifar100, vggnet, …) then hit the "Model is a DAG"
/// bail and scored `unknown` through the real competition runner, even though the
/// `beta-crown` CLI handled them fine. ALL prior DAG-guard coverage used
/// `beta-crown`, so this gap was invisible. The gate is now `!cli_branching_provided`
/// so `None` and `"auto"` behave identically. This test pins the competition path.
#[ntest::timeout(120000)]
#[test]
fn test_vnncomp_subcommand_dag_conv_auto_branching_no_bail() {
    let dir = tinyimagenet_dir();
    let onnx = dir.join("onnx/TinyImageNet_resnet_medium.onnx");
    let vnnlib =
        dir.join("vnnlib/TinyImageNet_resnet_medium_prop_idx_1126_sidx_4974_eps_0.0039.vnnlib");
    require_benchmark_file(&onnx);
    require_benchmark_file(&vnnlib);

    let results = NamedTempFile::new().expect("create temp results file");
    let output = run_ny(&[
        "vnncomp",
        "v1",
        "tinyimagenet_2024",
        onnx.to_str().expect("onnx path UTF-8"),
        vnnlib.to_str().expect("vnnlib path UTF-8"),
        results.path().to_str().expect("results path UTF-8"),
        // Small budget: the DAG bail (the bug) fires at model-load time in <0.1s, so a
        // short timeout still distinguishes "bailed" from "ran BaB then timed out".
        "8",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Model is a DAG"),
        "the `ny vnncomp` entry point must auto-select ReLU-splitting for DAG conv \
         models, not bail with the DAG branching guard.\nstdout: {stdout}\nstderr: {stderr}"
    );

    // A sound verdict must be written (timeout => `unknown`); never `error`, and never
    // an empty file from an early bail.
    let verdict_file = fs::read_to_string(results.path())
        .expect("vnncomp must write a verdict file")
        .trim()
        .to_string();
    // The first line is the VNN-COMP result token; for `sat` the SMT-LIB witness
    // follows on subsequent lines, so assert on the token only.
    let verdict = verdict_file.lines().next().unwrap_or("").trim();
    assert!(
        matches!(verdict, "unsat" | "sat" | "unknown" | "timeout"),
        "expected a sound VNN-COMP verdict, got {verdict_file:?}.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

/// Regression test for the cgan (BatchNorm + ConvTranspose generative model) crash.
///
/// cgan used to PANIC with `ndarray: index out of bounds` in the patches-mode
/// BatchNorm CROWN backward: that path indexes the bias vector by patches
/// output-neuron index, which is only valid when there is exactly one bias entry
/// per output neuron. For cgan's disjunctive multi-clause input split the spec/bias
/// layout differs, so the shorter bias vector was indexed out of bounds (SIGABRT).
/// The patches path now returns an error on that mismatch, so the verifier falls
/// back to the sound dense BatchNorm backward. The `ny vnncomp` runner must produce
/// a sound verdict, never crash.
#[ntest::timeout(120000)]
#[test]
fn test_vnncomp_cgan_batchnorm_patches_no_panic() {
    let dir = cgan_dir();
    let onnx = dir.join("onnx/cGAN_imgSz32_nCh_1.onnx");
    let vnnlib =
        dir.join("vnnlib/cGAN_imgSz32_nCh_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib");
    require_benchmark_file(&onnx);
    require_benchmark_file(&vnnlib);

    let results = NamedTempFile::new().expect("create temp results file");
    let output = run_ny(&[
        "vnncomp",
        "v1",
        "cgan_2023",
        onnx.to_str().expect("onnx path UTF-8"),
        vnnlib.to_str().expect("vnnlib path UTF-8"),
        results.path().to_str().expect("results path UTF-8"),
        "15",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("index out of bounds") && !stderr.contains("panicked"),
        "patches-mode BatchNorm backward must fall back to dense, not panic on the \
         spec/bias layout mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let verdict_file = fs::read_to_string(results.path())
        .expect("vnncomp must write a verdict file")
        .trim()
        .to_string();
    // The first line is the VNN-COMP result token; for `sat` the SMT-LIB witness
    // follows on subsequent lines, so assert on the token only.
    let verdict = verdict_file.lines().next().unwrap_or("").trim();
    assert!(
        matches!(verdict, "unsat" | "sat" | "unknown" | "timeout"),
        "expected a sound VNN-COMP verdict, got {verdict_file:?}.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

fn write_nn4sys_ibp_enhancement_preset_4372() -> NamedTempFile {
    let base_preset = workspace_root().join("configs/vnncomp25/nn4sys.yaml");
    let contents =
        fs::read_to_string(&base_preset).expect("should read base nn4sys preset for test");
    let patched = if contents.contains("      ibp_enhancement: true") {
        contents
    } else {
        contents.replace(
            "      sb_coeff_thresh: 0.1\n      reorder_bab: true",
            "      sb_coeff_thresh: 0.1\n      ibp_enhancement: true\n      reorder_bab: true",
        )
    };
    assert!(
        patched.contains("      ibp_enhancement: true"),
        "test preset rewrite must enable input_split.ibp_enhancement"
    );

    let mut temp = NamedTempFile::new().expect("should create temporary nn4sys preset");
    temp.write_all(patched.as_bytes())
        .expect("should write temporary nn4sys preset");
    temp
}

fn parse_success_json(output: &std::process::Output, command_label: &str) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "{command_label} exited with failure.\nstdout: {stdout}\nstderr: {stderr}"
    );

    parse_json_output(output, command_label)
}

fn assert_valid_verifier_exit(output: &std::process::Output, command_label: &str) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(
        VALID_EXIT_CODES.contains(&exit_code),
        "{command_label} exited with unexpected code {exit_code}.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

/// Smoke test VNN-COMP audit on a real CNN benchmark category.
///
/// Validates that `ny vnncomp-audit` can discover and load cifar10_resnet
/// benchmark assets (real ONNX + VNN-LIB files), not just synthetic unit models.
#[ntest::timeout(60000)]
#[test]
fn test_vnncomp_audit_cifar10_resnet_supported() {
    let category_dir = cifar10_resnet_dir();
    require_benchmark_file(&category_dir);

    let output = run_ny(&[
        "vnncomp-audit",
        "--year",
        "2021",
        "--category",
        "cifar10_resnet",
        "--json",
    ]);
    let json = parse_success_json(&output, "vnncomp-audit");

    assert_eq!(
        json["year"].as_u64(),
        Some(2021),
        "audit summary year must match request"
    );
    assert_eq!(
        json["total_categories"].as_u64(),
        Some(1),
        "category filter should isolate exactly one category"
    );
    assert_eq!(
        json["supported"].as_u64(),
        Some(1),
        "cifar10_resnet must be marked supported by model+property loading"
    );
    assert_eq!(
        json["unsupported"].as_u64(),
        Some(0),
        "cifar10_resnet should not be reported unsupported"
    );

    let categories = json["categories"]
        .as_array()
        .expect("categories must be an array");
    assert_eq!(
        categories.len(),
        1,
        "category filter must return one category result"
    );
    let category = &categories[0];
    assert_eq!(
        category["name"].as_str(),
        Some("cifar10_resnet"),
        "unexpected category in filtered audit output"
    );
    assert_eq!(
        category["status"].as_str(),
        Some("Supported"),
        "cifar10_resnet should load as supported"
    );
    assert!(
        category["instance_count"].as_u64().unwrap_or(0) >= 1,
        "cifar10_resnet should report non-zero instances"
    );
    assert_eq!(
        category["sample_model"].as_str(),
        Some("resnet_2b.onnx"),
        "expected first sample model from cifar10_resnet instances.csv"
    );
    let sample_property = category["sample_property"]
        .as_str()
        .expect("sample_property must be a string");
    assert!(
        sample_property.ends_with(".vnnlib"),
        "expected vnnlib sample property, got: {sample_property}"
    );
}

/// Real ResNet beta-crown invocation must AUTO-SELECT a DAG-capable branching mode.
///
/// (Previously this test asserted the CLI bailed with "Model is a DAG" unless the
/// user passed `--branching input|relu`. That predates model-class auto-branching:
/// with `--branching` defaulting to `auto`, a conv DAG ResNet now auto-selects
/// ReLU/kFSB splitting and runs BaB. The bail must NOT appear. SOUND: branching
/// choice never changes a verdict.)
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_cifar10_resnet_auto_selects_dag_branching() {
    let category_dir = cifar10_resnet_dir();
    let model_path = category_dir.join("onnx/resnet_2b.onnx");
    let property_path = category_dir
        .join("vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--timeout",
        "20",
        "--max-domains",
        "20",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("Model is a DAG"),
        "auto-branching must handle the ResNet DAG without the explicit-branching bail.\nstdout: {stdout}\nstderr: {stderr}"
    );
    // A status must be produced (BaB ran); timeout/unknown are acceptable for a 20s budget.
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("expected JSON output from auto-branched ResNet run: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    assert!(
        json.get("status").is_some(),
        "expected a status field from the auto-branched ResNet run: {json}"
    );
}

/// Real lsnc_relu DAG benchmark must honor preset input splitting without a redundant
/// `--branching input` override on the CLI.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_lsnc_relu_preset_honors_input_split() {
    let category_dir = lsnc_relu_dir();
    let model_path = category_dir.join("onnx/relu_quadrotor2d_state.onnx");
    let property_path = category_dir.join("vnnlib/quadrotor2d_state_0.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/lsnc_relu.yaml");
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
        "1",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Model is a DAG"),
        "preset-driven lsnc_relu invocation should not trip the DAG branching guard.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON output for beta-crown lsnc_relu preset: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Regression for #3870: the GPU-BaB lsnc sidecar must not abort during the
/// opportunistic graph PGD phase with a batched shape mismatch.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_lsnc_relu_gpu_bab_sidecar_avoids_graph_pgd_shape_mismatch_3870() {
    let category_dir = lsnc_relu_dir();
    let model_path = category_dir.join("onnx/relu_quadrotor2d_state.onnx");
    let property_path = category_dir.join("vnnlib/quadrotor2d_state_8.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/lsnc_relu_gpu_bab.yaml");
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
        "1",
        "--gpu-bab",
        "--json",
    ]);

    assert_valid_verifier_exit(&output, "beta-crown lsnc_relu GPU-BaB sidecar");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "lsnc_relu GPU-BaB sidecar should not abort from graph PGD batching.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json = parse_json_output(&output, "beta-crown lsnc_relu GPU-BaB sidecar");
    assert!(
        json["status"].is_string(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Real LinearizeNN DAG benchmark must honor preset input splitting without a
/// redundant `--branching input` override on the CLI.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_linearizenn_preset_honors_input_split() {
    let category_dir = linearizenn_dir();
    let model_path = category_dir.join("onnx/AllInOne_10_10.onnx");
    let property_path = category_dir.join("vnnlib/prop_10_10.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/linearizenn_2024.yaml");
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
        "1",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Model is a DAG"),
        "preset-driven linearizenn invocation should not trip the DAG branching guard.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON output for beta-crown linearizenn preset: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Real nn4sys DAG benchmark must honor preset input splitting without a
/// redundant `--branching input` override on the CLI.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_preset_honors_input_split() {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/pensieve_small_simple.onnx");
    let property_path = category_dir.join("vnnlib/pensieve_simple_0.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/nn4sys.yaml");
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
        "1",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Model is a DAG"),
        "preset-driven nn4sys invocation should not trip the DAG branching guard.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON output for beta-crown nn4sys preset: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Real nn4sys parallel DAG benchmark must not fail the Concat IBP path when
/// branch topologies disagree on whether they retain the batch dimension.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_parallel_preset_avoids_concat_shape_mismatch() {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/pensieve_small_parallel.onnx");
    let property_path = category_dir.join("vnnlib/pensieve_parallel_4.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/nn4sys.yaml");
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
        "1",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "parallel nn4sys invocation should not fail at Concat shape validation.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!(
            "failed to parse JSON output for beta-crown nn4sys parallel preset: {e}\nstdout: {stdout}\nstderr: {stderr}"
        )
    });
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Regression for #4372: enabling input-split IBP enhancement on the real nn4sys
/// parallel pensieve row must degrade to a normal verification result instead of
/// hard-failing with the enhancement-only shape-mismatch path.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_parallel_ibp_enhancement_degrades_without_shape_error_4372() {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/pensieve_small_parallel.onnx");
    let property_path = category_dir.join("vnnlib/pensieve_parallel_4.vnnlib");
    let preset = write_nn4sys_ibp_enhancement_preset_4372();
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset.path().to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "1",
        "--max-domains",
        "1",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "nn4sys ibp_enhancement pensieve lane should degrade to the plain path instead of hard-failing.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert_valid_verifier_exit(&output, "beta-crown nn4sys ibp_enhancement pensieve");

    let json = parse_json_output(&output, "beta-crown nn4sys ibp_enhancement pensieve");
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Regression for #4372: the real nn4sys benchmark split path must not hard-fail
/// the reorder-bab child prescreen when `ibp_enhancement=true`.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_parallel_ibp_enhancement_split_path_degrades_without_shape_error_4372() {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/pensieve_small_parallel.onnx");
    let property_path = category_dir.join("vnnlib/pensieve_parallel_4.vnnlib");
    let preset = write_nn4sys_ibp_enhancement_preset_4372();
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset.path().to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "5",
        "--max-domains",
        "2",
        "--no-alpha",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("Shape mismatch") && !stderr.contains("Shape mismatch"),
        "nn4sys ibp_enhancement split path should degrade to the plain path instead of hard-failing.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert_valid_verifier_exit(
        &output,
        "beta-crown nn4sys ibp_enhancement pensieve split path",
    );

    let json = parse_json_output(
        &output,
        "beta-crown nn4sys ibp_enhancement pensieve split path",
    );
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Regression for #4372: the real nn4sys disjunctive `mscn_128d` row must not
/// hard-fail the enhancement-only Slice/IBP path when `ibp_enhancement=true`.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_disjunctive_ibp_enhancement_degrades_without_slice_error_4372() {
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/mscn_128d.onnx");
    let property_path = category_dir.join("vnnlib/cardinality_0_1_128.vnnlib");
    let preset = write_nn4sys_ibp_enhancement_preset_4372();
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset.path().to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "5",
        "--max-domains",
        "1",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("empty after clamping") && !stderr.contains("empty after clamping"),
        "nn4sys ibp_enhancement disjunctive lane should degrade to the plain path instead of hard-failing.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert_valid_verifier_exit(&output, "beta-crown nn4sys ibp_enhancement disjunctive");

    let json = parse_json_output(&output, "beta-crown nn4sys ibp_enhancement disjunctive");
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Regression for #4372: the real nn4sys disjunctive split path must not
/// hard-fail the reorder-bab child prescreen when `ibp_enhancement=true`.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_nn4sys_disjunctive_ibp_enhancement_split_path_degrades_without_slice_error_4372()
{
    let category_dir = nn4sys_dir();
    let model_path = category_dir.join("onnx/mscn_128d.onnx");
    let property_path = category_dir.join("vnnlib/cardinality_0_1_128.vnnlib");
    let preset = write_nn4sys_ibp_enhancement_preset_4372();
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);

    let output = run_ny(&[
        "beta-crown",
        model_path.to_str().expect("model path must be UTF-8"),
        "--property",
        property_path.to_str().expect("property path must be UTF-8"),
        "--preset",
        preset.path().to_str().expect("preset path must be UTF-8"),
        "--timeout",
        "5",
        "--max-domains",
        "2",
        "--json",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("empty after clamping") && !stderr.contains("empty after clamping"),
        "nn4sys ibp_enhancement disjunctive split path should degrade to the plain path instead of hard-failing.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert_valid_verifier_exit(
        &output,
        "beta-crown nn4sys ibp_enhancement disjunctive split path",
    );

    let json = parse_json_output(
        &output,
        "beta-crown nn4sys ibp_enhancement disjunctive split path",
    );
    assert!(
        json.get("status").is_some(),
        "expected beta-crown JSON output to include a status field: {json}"
    );
}

/// Preset boolean flags (relaxed_clip, pgd_attack) must flow through to the config
/// and not be overwritten by CLI boolean defaults.
///
/// Regression test for the bug where `#[arg(long, default_value_t = false)]` booleans
/// unconditionally overwrote preset-enabled features. Part of #3218.
#[ntest::timeout(60000)]
#[test]
fn test_preset_boolean_flags_not_overwritten_by_cli_defaults() {
    let category_dir = lsnc_relu_dir();
    let model_path = category_dir.join("onnx/relu_quadrotor2d_state.onnx");
    let property_path = category_dir.join("vnnlib/quadrotor2d_state_0.vnnlib");
    let preset_path = workspace_root().join("configs/vnncomp25/lsnc_relu.yaml");
    require_benchmark_file(&model_path);
    require_benchmark_file(&property_path);
    require_benchmark_file(&preset_path);

    // Run with preset only (no explicit --relaxed-clip or --pgd-attack on CLI).
    // The lsnc_relu preset has bab.clip.relaxed=true and attack.pgd_order=before,
    // so both should be reflected in the config display.
    let output = run_ny(&[
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
    ]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Combine both streams since Config line may go to either
    let combined = format!("{stdout}\n{stderr}");

    assert!(
        combined.contains("relaxed_clip=true"),
        "preset bab.clip.relaxed=true must appear as relaxed_clip=true in config, \
         not be overwritten by CLI default=false.\noutput: {combined}"
    );
    assert!(
        combined.contains("pgd_attack=true"),
        "preset attack.pgd_order=before must appear as pgd_attack=true in config, \
         not be overwritten by CLI default=false.\noutput: {combined}"
    );
}
