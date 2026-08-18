// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `ny beta-crown` CLI subcommand.
//!
//! These tests exercise the full end-to-end path: CLI binary → model loading →
//! BaB verification → JSON output. This is the competition entry path used by
//! `vnncomp_scripts/run_instance.sh`.
//!
//! Part of #2604.

use ny_test_utils::{require_model, test_models_dir, workspace_root};
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

/// Get the path to the ny binary.
///
/// Resolution order:
/// 1. `CARGO_BIN_EXE_ny` (set by cargo for integration tests)
/// 2. Most recently modified `target/worker_*/debug/ny` (per-worker builds)
/// 3. `target/debug/ny`
/// 4. `target/release/ny`
/// 5. System `ny` binary
fn ny_binary() -> String {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ny") {
        return bin;
    }

    let workspace = workspace_root();

    // Check per-worker target directories (most recently modified first).
    // The cargo wrapper uses target/worker_N/ for concurrent builds (#5590).
    if let Ok(entries) = std::fs::read_dir(workspace.join("target")) {
        let mut worker_bins: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("worker_"))
            })
            .filter_map(|e| {
                let alias = e.path().join("debug/ny");
                let alias_mtime = alias.metadata().ok()?.modified().ok();
                let legacy = e.path().join("debug/ny");
                let legacy_mtime = legacy.metadata().ok()?.modified().ok();
                match (alias_mtime, legacy_mtime) {
                    (Some(mtime), _) => Some((alias, mtime)),
                    (None, Some(mtime)) => Some((legacy, mtime)),
                    (None, None) => None,
                }
            })
            .collect();
        worker_bins.sort_by_key(|bin| std::cmp::Reverse(bin.1));
        if let Some((bin, _)) = worker_bins.first() {
            return bin.to_string_lossy().to_string();
        }
    }

    let debug_alias_bin = workspace.join("target/debug/ny");
    let release_alias_bin = workspace.join("target/release/ny");
    let debug_bin = workspace.join("target/debug/ny");
    let release_bin = workspace.join("target/release/ny");

    if debug_alias_bin.exists() {
        debug_alias_bin.to_string_lossy().to_string()
    } else if release_alias_bin.exists() {
        release_alias_bin.to_string_lossy().to_string()
    } else if debug_bin.exists() {
        debug_bin.to_string_lossy().to_string()
    } else if release_bin.exists() {
        release_bin.to_string_lossy().to_string()
    } else {
        "ny".to_string()
    }
}

const VALID_STATUSES: [&str; 5] = [
    "verified",
    "violated",
    "potential_violation",
    "unknown",
    "timeout",
];

/// Valid exit codes for `ny beta-crown`:
/// 0 = verified, 1 = violated, 2 = unknown, 3 = timeout.
/// All are legitimate verification outcomes, not errors.
const VALID_EXIT_CODES: [i32; 4] = [0, 1, 2, 3];

/// Run `ny beta-crown` with the given args and return parsed JSON output,
/// asserting that the process code agrees with the machine-readable verdict.
fn run_beta_crown_json(args: &[&str]) -> serde_json::Value {
    run_beta_crown_json_env(args, &[])
}

/// Like [`run_beta_crown_json`], but with extra environment variables set on
/// the child process (e.g. `RAYON_NUM_THREADS` to pin BaB parallelism).
fn run_beta_crown_json_env(args: &[&str], envs: &[(&str, &str)]) -> serde_json::Value {
    let output = Command::new(ny_binary())
        .args(args)
        .envs(envs.iter().copied())
        .output()
        .expect("failed to execute ny binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let exit_code = output.status.code().unwrap_or(-1);
    assert!(
        VALID_EXIT_CODES.contains(&exit_code),
        "ny beta-crown exited with unexpected code {exit_code}.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON output: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    let status = json["status"]
        .as_str()
        .unwrap_or_else(|| panic!("JSON outcome lacks status: {json}"));
    let expected_exit_code = match status {
        "verified" => 0,
        "violated" => 1,
        "potential_violation" | "unknown" => 2,
        "timeout" => 3,
        other => panic!("unexpected beta-crown status {other:?}: {json}"),
    };
    assert_eq!(
        exit_code, expected_exit_code,
        "beta-crown status/exit mismatch.\nstdout: {stdout}\nstderr: {stderr}"
    );
    json
}

/// Assert the JSON contains a valid status and core numeric fields.
fn assert_core_json_fields(json: &serde_json::Value) {
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    assert!(
        VALID_STATUSES.contains(&status),
        "unexpected status value: {status}"
    );
    assert!(
        json["domains_explored"].is_number(),
        "JSON must contain 'domains_explored' number"
    );
    assert!(
        json["time_elapsed_s"].is_number(),
        "JSON must contain 'time_elapsed_s' number"
    );
}

/// End-to-end: `ny beta-crown` with ACAS Xu NNet model and VNN-LIB property.
///
/// Uses acasxu_1_1.nnet (5 inputs, 5 outputs, 6 hidden layers) with property 2
/// (relational constraints: Y_1..Y_4 <= Y_0). This is the standard VNN-COMP
/// verification task.
///
/// Keep the domain cap well below the wall-clock timeout: this is a functional
/// CLI/JSON contract test, not a throughput benchmark. A cap of two still
/// exercises a real BaB split, while the former cap of 200 raced the 30-second
/// deadline and nondeterministically returned either `unknown` or `timeout`.
#[ntest::timeout(90000)]
#[test]
fn test_beta_crown_nnet_with_vnnlib_json() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    require_model(&model_path);
    let property_path = test_models_dir().join("acasxu_prop2.vnnlib");
    require_model(&property_path);

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "45",
        "--max-domains",
        "2",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
    assert_eq!(
        json["status"]
            .as_str()
            .expect("JSON output must contain 'status' string field"),
        "unknown",
        "the bounded ACAS Xu run should stop inconclusively at its domain cap"
    );
    assert!(
        json["reason"]
            .as_str()
            .is_some_and(|reason| reason.starts_with("Domain limit 2")),
        "the functional test must finish through the domain-limit path, not hide a timeout: {json}"
    );
    assert_eq!(
        json["domains_explored"].as_u64(),
        Some(2),
        "the bounded run must consume exactly its two-domain cap: {json}"
    );
    assert!(
        json["max_depth_reached"]
            .as_u64()
            .is_some_and(|depth| depth >= 1),
        "the capped run must exercise at least one real BaB split: {json}"
    );
    assert!(
        json["threshold"].is_number(),
        "JSON must contain 'threshold' number"
    );
    assert!(
        json["domains_verified"].is_number(),
        "JSON must contain 'domains_verified' number"
    );
}

/// End-to-end: `ny beta-crown` with ONNX model and inline VNN-LIB property.
///
/// Uses simple_mlp.onnx (2 inputs, 2 outputs) with a property written to a
/// temp file. Verifies the ONNX loading → sequential BaB → JSON output pipeline.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_onnx_with_inline_vnnlib_json() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    // VNN-LIB property: inputs in [-1, 1], output constraint Y_0 <= Y_1.
    // The CLI runs end-to-end regardless of verification result.
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
";

    let mut vnnlib_file =
        NamedTempFile::new().expect("failed to create temp file for VNN-LIB property");
    vnnlib_file
        .write_all(vnnlib_content.as_bytes())
        .expect("failed to write VNN-LIB property");

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        vnnlib_file.path().to_str().expect("temp path is UTF-8"),
        "--timeout",
        "10",
        "--max-domains",
        "100",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
}

/// JSON schema validation: all expected fields present with correct types.
///
/// Validates every field in the output schema from `commands/beta_crown/output.rs`.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_json_schema_complete() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    require_model(&model_path);
    let property_path = test_models_dir().join("acasxu_prop2.vnnlib");
    require_model(&property_path);

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "10",
        "--max-domains",
        "50",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    // Exhaustive field presence check (all 11 fields from output.rs)
    for field in [
        "status",
        "reason",
        "counterexample",
        "property_file",
        "threshold",
        "domains_explored",
        "domains_verified",
        "cuts_generated",
        "max_depth_reached",
        "time_elapsed_s",
        "output_bound_width",
    ] {
        assert!(
            json.get(field).is_some(),
            "JSON output missing expected field '{field}'. Got: {json}"
        );
    }

    assert_core_json_fields(&json);
    assert_numeric_fields(&json);
    assert_nullable_fields(&json);
}

/// Falsifiable property check: catches a regressor that always reports "verified".
///
/// The constraint `Y_0 <= 100` is satisfiable for this tiny network and input domain,
/// so returning `verified` would be unsound.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_falsifiable_property_is_not_verified() {
    let model_path = test_models_dir().join("crossing_relu.nnet");
    require_model(&model_path);

    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 100.0))
";

    let mut vnnlib_file =
        NamedTempFile::new().expect("failed to create temp file for VNN-LIB property");
    vnnlib_file
        .write_all(vnnlib_content.as_bytes())
        .expect("failed to write VNN-LIB property");

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        vnnlib_file.path().to_str().expect("temp path is UTF-8"),
        "--timeout",
        "5",
        "--max-domains",
        "20",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    // After #3678, PotentialViolation is either confirmed to Violated (concrete
    // counterexample found) or downgraded to Unknown. The easy property Y_0 <= 100
    // should be confirmed as Violated by the post-BaB sampling attack.
    assert!(
        status == "violated" || status == "unknown",
        "falsifiable property should be confirmed violated or downgraded to unknown, got: {status}"
    );
}

/// End-to-end CNN: Conv2d → ReLU → MaxPool → Flatten → Linear with VNN-LIB property.
///
/// Exercises the full CNN verification pipeline: ONNX model with Conv2d layer →
/// sequential bound propagation through convolution → BaB verification → JSON
/// output. This is the first integration test that exercises Conv2d through
/// the CLI entry point.
///
/// Architecture: Conv(3x3, pad=1, 1→4ch) → ReLU → MaxPool(2x2) → Flatten → Linear(64→2)
/// Input: (1, 1, 8, 8), Output: (1, 2)
/// Property: adversarial robustness (unsafe region Y_0 <= Y_1 unreachable;
/// class 0 wins by ~0.14 at the box center) around center point with eps=0.01
///
/// Part of #2665.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_cnn_with_flatten_verified() {
    let model_path = test_models_dir().join("cnn_with_flatten.onnx");
    require_model(&model_path);
    let property_path = test_models_dir().join("cnn_with_flatten.vnnlib");
    require_model(&property_path);

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "30",
        "--max-domains",
        "100",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    assert_eq!(
        status, "verified",
        "cnn_with_flatten should verify: Conv2d→ReLU→MaxPool→Flatten→Linear with tight epsilon"
    );
    assert!(
        json["domains_explored"]
            .as_u64()
            .expect("domains_explored must be a number")
            >= 1,
        "must explore at least 1 domain"
    );
    // The verified objective is one-sided (lower bound on the spec margin),
    // so no finite two-sided output width is guaranteed: the field is null
    // when the upper side is unbounded, and finite when bounds are reported.
    assert!(
        json["output_bound_width"].is_null()
            || json["output_bound_width"]
                .as_f64()
                .is_some_and(f64::is_finite),
        "output_bound_width must be null or finite, got: {}",
        json["output_bound_width"]
    );
}

/// End-to-end CNN alpha-CROWN with Patches mode via auto graph routing.
///
/// Exercises the patches-mode pipeline added by W2 commits 38-41 (#3218):
/// 1. Conv2d detection triggers auto-routing to GraphNetwork (mod.rs:267-299)
/// 2. Alpha-CROWN precheck is skipped for CNN models (Conv2d fallback)
/// 3. Graph engine runs patches-mode CROWN backward (crown.rs CrownBounds)
/// 4. Alpha optimization tightens bounds on ReLU layers via patches
/// 5. BaB completes verification
///
/// Without `--no-alpha`, the Conv2d model auto-routes to GraphNetwork with
/// ReLU splitting. This is the VNN-COMP competition path for CNN categories.
///
/// Architecture: Conv(3x3, pad=1, 1→4ch) → ReLU → MaxPool(2x2) → Flatten → Linear(64→2)
/// Input: (1, 1, 8, 8), Output: (1, 2)
///
/// Part of #3218.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_cnn_alpha_crown_patches_mode() {
    let model_path = test_models_dir().join("cnn_with_flatten.onnx");
    require_model(&model_path);
    let property_path = test_models_dir().join("cnn_with_flatten.vnnlib");
    require_model(&property_path);

    // No --no-alpha: alpha-CROWN enabled, Conv2d auto-routes to graph+patches
    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "30",
        "--max-domains",
        "100",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    assert!(
        VALID_STATUSES.contains(&status),
        "CNN alpha-CROWN patches must produce a valid status, got: {status}"
    );

    // The key assertion: alpha-CROWN with patches mode must not panic, produce
    // NaN, or fail. It should either verify or explore domains without error.
    let domains = json["domains_explored"]
        .as_u64()
        .expect("domains_explored must be a number");
    assert!(
        domains >= 1,
        "alpha-CROWN patches mode must explore at least 1 domain"
    );

    // Guard against NaN from patches-mode CROWN backward
    if let Some(width) = json["output_bound_width"].as_f64() {
        assert!(
            width.is_finite(),
            "output_bound_width must be finite with patches mode, got: {width}"
        );
    }
    assert!(
        json["time_elapsed_s"]
            .as_f64()
            .expect("time_elapsed_s must be a number")
            .is_finite(),
        "time_elapsed_s must be finite"
    );
}

/// End-to-end CNN BaB loop: Conv2d → ReLU → MaxPool → Flatten → Gemm exercises
/// branch-and-bound splitting on a CNN model.
///
/// This model has a harder property than cnn_with_flatten, so BaB explores
/// multiple domains without resolving. The test validates that the BaB loop
/// runs correctly through Conv2d layers without panicking or producing NaN.
///
/// Architecture: Conv(3x3, pad=1, 1→4ch) → ReLU → MaxPool(2x2) → Flatten → Gemm(64→2)
/// Input: (1, 1, 8, 8), Output: (1, 2)
///
/// Part of #2665.
#[ntest::timeout(60000)]
#[test]
fn test_beta_crown_cnn_maxpool_bab_loop() {
    let model_path = test_models_dir().join("test_cnn_maxpool.onnx");
    require_model(&model_path);
    let property_path = test_models_dir().join("test_cnn_maxpool.vnnlib");
    require_model(&property_path);

    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "15",
        "--max-domains",
        "100",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--json",
    ]);

    assert_core_json_fields(&json);
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    // This CNN model exercises BaB splitting. It may verify, hit domain limit,
    // or time out — all are valid outcomes. The key assertion is that BaB
    // runs without panicking on a Conv2d model and explores multiple domains.
    assert!(
        VALID_STATUSES.contains(&status),
        "CNN BaB must produce a valid status, got: {status}"
    );
    let domains = json["domains_explored"]
        .as_u64()
        .expect("domains_explored must be a number");
    assert!(
        domains >= 2,
        "BaB on CNN should explore multiple domains (got {domains}), confirming the splitting loop works through Conv2d"
    );
    // Guard against NaN contamination: all numeric fields must be finite
    if let Some(width) = json["output_bound_width"].as_f64() {
        assert!(
            width.is_finite(),
            "output_bound_width must be finite (no NaN/Inf), got: {width}"
        );
    }
    assert!(
        json["time_elapsed_s"]
            .as_f64()
            .expect("time_elapsed_s must be a number")
            .is_finite(),
        "time_elapsed_s must be finite"
    );
}

/// Assert all non-nullable numeric fields have number type.
fn assert_numeric_fields(json: &serde_json::Value) {
    for field in [
        "threshold",
        "domains_explored",
        "domains_verified",
        "cuts_generated",
        "max_depth_reached",
        "time_elapsed_s",
    ] {
        assert!(
            json[field].is_number(),
            "'{field}' must be a number, got: {}",
            json[field]
        );
    }
}

/// Assert nullable fields are either null or the expected type.
fn assert_nullable_fields(json: &serde_json::Value) {
    assert!(
        json["reason"].is_null() || json["reason"].is_string(),
        "'reason' must be null or string"
    );
    assert!(
        json["counterexample"].is_null() || json["counterexample"].is_object(),
        "'counterexample' must be null or object"
    );
}

/// Regression test for #2800: contradictory VNN-LIB bounds (lower > upper) must
/// produce a structured error, NOT a panic/crash from `Bound::new`.
///
/// Previously, `build_verification_spec` called `Bound::new(l, u)` on unvalidated
/// VNN-LIB bounds, panicking on inverted intervals. After the fix, the path uses
/// `Bound::try_new_allow_infinite` which returns a contextual error.
#[ntest::timeout(10000)]
#[test]
fn test_verify_contradictory_vnnlib_returns_error_not_panic_2800() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    // Contradictory: X_0 >= 1.0 AND X_0 <= 0.0 → lower=1.0 > upper=0.0
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 1.0))
(assert (<= X_0 0.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
";

    let mut vnnlib_file =
        NamedTempFile::new().expect("failed to create temp file for VNN-LIB property");
    vnnlib_file
        .write_all(vnnlib_content.as_bytes())
        .expect("failed to write VNN-LIB property");

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().expect("model path is UTF-8"),
            "--property",
            vnnlib_file.path().to_str().expect("temp path is UTF-8"),
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("failed to execute ny binary");

    // The process must NOT have been killed by a signal (panic/abort).
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "verify should return an error, not panic. stderr: {stderr}"
    );
    // The process should exit with failure (non-zero) since the spec is invalid.
    assert!(
        !output.status.success(),
        "verify should fail on contradictory VNN-LIB bounds, but exited successfully"
    );
}

/// Regression test for #2800: non-finite VNN-LIB input bounds must surface as
/// structured errors, not panic via bound construction.
#[ntest::timeout(10000)]
#[test]
fn test_verify_nan_vnnlib_bound_returns_error_not_panic_2800() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    // NaN lower bound for X_0 should be rejected during verification spec build.
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 NaN))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
";

    let mut vnnlib_file =
        NamedTempFile::new().expect("failed to create temp file for VNN-LIB property");
    vnnlib_file
        .write_all(vnnlib_content.as_bytes())
        .expect("failed to write VNN-LIB property");

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().expect("model path is UTF-8"),
            "--property",
            vnnlib_file.path().to_str().expect("temp path is UTF-8"),
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("failed to execute ny binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stderr.contains("panicked at"),
        "verify should return an error, not panic. stderr: {stderr}"
    );
    assert!(
        !output.status.success(),
        "verify should fail on non-finite VNN-LIB bounds, but exited successfully"
    );
    // With --json, errors are in stdout JSON, not stderr.
    // Parse JSON and check the message field for VNN-LIB bound context.
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("failed to parse JSON error output: {e}\nstdout: {stdout}\nstderr: {stderr}")
    });
    let message = json["message"].as_str().unwrap_or("");
    assert!(
        message.contains("X_0") && message.contains("invalid"),
        "JSON error message should identify invalid VNN-LIB bound. message: {message}"
    );
}

/// End-to-end timeout enforcement: `--timeout 2` must cause the process to
/// finish within a bounded wall-clock window and report timeout/unknown status.
///
/// This is the end-to-end regression test for #3328. The CROWN backward pass
/// and CROWN-IBP collection now check the deadline between layers, so even the
/// initial bounds computation respects the timeout. Previously, the initial
/// CROWN pass ran to completion regardless of the timeout flag.
///
/// Uses ACAS Xu (6 hidden layers, relational property with 4 safety
/// constraints) which requires BaB and cannot verify in 2 seconds on a single
/// core. BaB is pinned to one rayon worker (`RAYON_NUM_THREADS=1`) so the
/// deadline genuinely expires: on a many-core box the parallel BaB verifies
/// this instance in under a second (~6.5s of CPU across 20 workers), which
/// tests nothing about the deadline. `--no-certificate` keeps the test scoped
/// to BaB deadline enforcement: the post-verdict certificate pass has its own
/// deadline gates and its own regression test
/// (`test_beta_crown_certificate_pass_honors_timeout` below).
///
/// Part of #3328.
#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_timeout_enforcement_3328() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    require_model(&model_path);
    let property_path = test_models_dir().join("acasxu_prop2.vnnlib");
    require_model(&property_path);

    let start = std::time::Instant::now();
    let json = run_beta_crown_json_env(
        &[
            "beta-crown",
            model_path.to_str().expect("model path is UTF-8"),
            "--property",
            property_path.to_str().expect("property path is UTF-8"),
            "--timeout",
            "2",
            "--max-domains",
            "50000",
            "--no-alpha",
            "--complete-verifier",
            "bab",
            "--no-certificate",
            "--json",
        ],
        &[("RAYON_NUM_THREADS", "1")],
    );
    let wall_time = start.elapsed();

    assert_core_json_fields(&json);

    // Deadline enforcement: the process must not massively overshoot the timeout.
    // Allow generous margin (3x) for process startup, model loading, and one
    // layer backward pass that was in-flight when the deadline hit.
    let wall_secs = wall_time.as_secs_f64();
    assert!(
        wall_secs < 10.0,
        "wall time {wall_secs:.1}s far exceeds --timeout 2: deadline enforcement may not be working"
    );

    // Status must be timeout or unknown — both are valid when deadline fires.
    // "timeout" = BaB loop hit deadline directly.
    // "unknown" = per-constraint aggregation mapped timeout to unknown.
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    assert!(
        status == "timeout" || status == "unknown",
        "with --timeout 2 on ACAS Xu prop2, expected timeout or unknown, got: {status}"
    );

    // time_elapsed_s in JSON must be finite and consistent with wall time.
    let reported_time = json["time_elapsed_s"]
        .as_f64()
        .expect("time_elapsed_s must be a number");
    assert!(
        reported_time.is_finite() && reported_time > 0.0,
        "time_elapsed_s must be finite and positive, got: {reported_time}"
    );
    // Reported time should not exceed wall time by more than 1s (measurement jitter).
    assert!(
        reported_time <= wall_secs + 1.0,
        "reported time {reported_time:.2}s exceeds wall time {wall_secs:.2}s by >1s — timer bug?"
    );
    // Reported time should be at least 0.5s — catches premature exits misclassified
    // as timeout (e.g., model loading error returning "unknown" with near-zero time).
    assert!(
        reported_time >= 0.5,
        "reported time {reported_time:.2}s suspiciously low for --timeout 2 — premature exit?"
    );
}

/// The post-verdict certificate pass must also honor `--timeout` (#3328).
///
/// On a many-core box, parallel BaB verifies ACAS Xu 1_1 / prop2 well inside a
/// 2s budget (~0.9s wall). The Verified verdict then enters
/// `maybe_emit_certificate` → `ny_cert::crown_deep::DeepReluProblem::certify`,
/// an exact-BigRational CROWN pass that — before the fix — had no deadline
/// threading and effectively never terminated on this net: unstable-ReLU
/// envelope slopes make the interned rationals grow to thousands of bits
/// across 6 layers, and every `Rat` op re-hashes a `BigRational` via
/// num-rational's continued-fraction recursive `Hash` (observed >1300
/// recursion frames, one BigInt division each, inside
/// `ny_cert::rational::intern`'s dedup HashMap insert) — >150s of 100% CPU
/// against `--timeout 2` with no output.
///
/// The pass is now deadline-bounded twice over: emission is skipped outright
/// when the remaining budget is under the 2s floor (the case here — BaB leaves
/// ~1s of the 2s budget), and an in-flight exact derivation aborts fail-closed
/// at coarse loop boundaries via the threaded absolute deadline. Either way
/// the CLI must return within the budget (+ startup grace) with the verdict
/// intact and NO sidecar written — a partial or unchecked certificate must
/// never appear. `--emit-certificate` (which force-enables emission) both pins
/// the sidecar path into a temp dir and proves the budget gate outranks even
/// an explicit emission request.
#[ntest::timeout(30000)]
#[test]
fn test_beta_crown_certificate_pass_honors_timeout() {
    let model_path = test_models_dir().join("acasxu_1_1.nnet");
    require_model(&model_path);
    let property_path = test_models_dir().join("acasxu_prop2.vnnlib");
    require_model(&property_path);
    let cert_dir = tempfile::tempdir().expect("temp dir for certificate sidecar");
    let cert_path = cert_dir.path().join("acasxu_1_1.cert.json");

    let start = std::time::Instant::now();
    // Identical to test_beta_crown_timeout_enforcement_3328 but with the
    // certificate pass ENABLED and full parallelism, so a fast Verified
    // verdict reaches the certificate pass with ~1s of the 2s budget
    // remaining — the pass must bail within that remainder, not run
    // unboundedly.
    let json = run_beta_crown_json(&[
        "beta-crown",
        model_path.to_str().expect("model path is UTF-8"),
        "--property",
        property_path.to_str().expect("property path is UTF-8"),
        "--timeout",
        "2",
        "--max-domains",
        "50000",
        "--no-alpha",
        "--complete-verifier",
        "bab",
        "--emit-certificate",
        cert_path.to_str().expect("temp cert path is UTF-8"),
        "--json",
    ]);
    let wall_secs = start.elapsed().as_secs_f64();

    assert_core_json_fields(&json);
    // Same wall-clock window as the BaB deadline test: generous margin over
    // --timeout 2 for process startup and model loading, but nowhere near the
    // unbounded (>150s) pre-fix overshoot.
    assert!(
        wall_secs < 10.0,
        "wall time {wall_secs:.1}s far exceeds --timeout 2: the certificate \
         pass must be bounded by the remaining wall-clock budget"
    );
    // assert_core_json_fields pinned status to a real verification outcome. On
    // a many-core box BaB decides this instance well inside the budget, so the
    // verdict must survive the skipped certificate untouched; a loaded box may
    // instead hit the BaB deadline first (timeout/unknown), which keeps the
    // contract too. Never an error/crash.
    let status = json["status"]
        .as_str()
        .expect("JSON output must contain 'status' string field");
    assert!(
        ["verified", "timeout", "unknown"].contains(&status),
        "with --timeout 2 on ACAS Xu prop2, expected verified (fast BaB, cert \
         skipped) or timeout/unknown (slow box), got: {status}"
    );
    // The certificate must be SKIPPED, never partially written: with ~1s of
    // the 2s budget left after BaB, emission is below the 2s floor (and any
    // in-flight derivation would abort at the deadline before writing).
    assert!(
        !cert_path.exists(),
        "no certificate sidecar may be written when the --timeout budget is \
         exhausted; found {}",
        cert_path.display()
    );
}
