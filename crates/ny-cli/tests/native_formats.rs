// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for native model format support.
//!
//! Tests that ny can load and verify models in various native formats:
//! - SafeTensors (.safetensors)
//! - PyTorch (.pt, .pth, .bin)
//! - GGUF (.gguf)
//!
//! ## Test Categories
//!
//! - **In-repo models**: Tests using `tests/models/` - always run
//! - **External models**: Tests using `models/` - require `external-models` feature
//!
//! To run external model tests:
//! ```bash
//! cargo test -p ny-cli --features external-models
//! ```

#[cfg(feature = "external-models")]
use ny_test_utils::{external_models_dir, require_external_model};
use ny_test_utils::{require_model, test_models_dir, workspace_root};
use std::process::Command;
use tempfile::tempdir;

/// Get the path to the ny binary (debug or release)
fn ny_binary() -> String {
    // When invoked under `cargo test`, Cargo exposes the compiled binary path via
    // an environment variable. Prefer it to avoid accidentally running a stale
    // `target/release/ny` from a different build profile.
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ny") {
        return bin;
    }

    let workspace = workspace_root();
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
        // Fallback to simple path
        "ny".to_string()
    }
}

// =============================================================================
// External Model Tests (require `external-models` feature)
// =============================================================================

/// Test that inspect command works with SafeTensors format
#[ntest::timeout(10000)]
#[test]
#[cfg(feature = "external-models")]
fn test_inspect_safetensors() {
    let model_path = external_models_dir().join("whisper-tiny/model.safetensors");
    require_external_model(&model_path);

    let output = Command::new(ny_binary())
        .args(["inspect", model_path.to_str().unwrap(), "--json"])
        .output()
        .expect("Failed to run ny inspect");

    assert!(
        output.status.success(),
        "ny inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("WhisperEncoder"),
        "Expected WhisperEncoder architecture"
    );
}

/// Test that inspect command works with PyTorch format
#[ntest::timeout(10000)]
#[test]
#[cfg(feature = "external-models")]
fn test_inspect_pytorch() {
    let model_path = external_models_dir().join("kokoro/kokoro-v1_0.pth");
    require_external_model(&model_path);

    let output = Command::new(ny_binary())
        .args(["inspect", model_path.to_str().unwrap(), "--json"])
        .output()
        .expect("Failed to run ny inspect");

    assert!(
        output.status.success(),
        "ny inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Kokoro"), "Expected Kokoro architecture");
}

/// Test that inspect command works with GGUF format
#[ntest::timeout(10000)]
#[test]
#[cfg(feature = "external-models")]
fn test_inspect_gguf() {
    let model_path = external_models_dir().join("gemma-2b-q4.gguf");
    require_external_model(&model_path);

    let output = Command::new(ny_binary())
        .args(["inspect", model_path.to_str().unwrap(), "--json"])
        .output()
        .expect("Failed to run ny inspect");

    assert!(
        output.status.success(),
        "ny inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // GGUF models load with detected architecture
    assert!(
        stdout.contains("parameters") || stdout.contains("Weight tensors"),
        "Expected model info in output"
    );
}

// =============================================================================
// In-Repo Model Tests (always run)
// =============================================================================

/// Test simple ONNX verification still works
#[ntest::timeout(10000)]
#[test]
fn test_verify_simple_mlp() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse JSON to check status more robustly
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");
    assert_eq!(
        v.get("status").and_then(|s| s.as_str()),
        Some("verified"),
        "Expected verified status, got: {}",
        stdout
    );
}

/// Test property_status is omitted when VNNLIB has no output constraints.
#[ntest::timeout(10000)]
#[test]
fn test_verify_property_status_absent_without_output_constraints() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let dir = tempdir().expect("Failed to create tempdir");
    let property_path = dir.path().join("no_output_constraints.vnnlib");
    let content = r#"
(vnnlib-version 1.0)
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
"#;
    std::fs::write(&property_path, content).expect("Failed to write VNNLIB file");

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--property",
            property_path.to_str().unwrap(),
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");
    let status = v
        .get("status")
        .and_then(|s| s.as_str())
        .expect("Expected status in JSON output");
    assert!(
        ["verified", "violated", "unknown", "timeout"].contains(&status),
        "Unexpected status '{}' in JSON output: {}",
        status,
        stdout
    );
    assert!(
        v.get("property_status").is_none(),
        "Expected property_status to be omitted when no output constraints, got: {}",
        stdout
    );
}

/// Test CROWN method verification
#[ntest::timeout(10000)]
#[test]
fn test_verify_crown_method() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "crown",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse JSON to check status more robustly
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");
    assert_eq!(
        v.get("status").and_then(|s| s.as_str()),
        Some("verified"),
        "Expected verified status, got: {}",
        stdout
    );
    assert_eq!(
        v.get("method").and_then(|s| s.as_str()),
        Some("crown"),
        "Expected crown method, got: {}",
        stdout
    );
}

/// Test layer benchmarks run correctly
#[ntest::timeout(120000)]
#[test]
fn test_bench_layer() {
    let output = Command::new(ny_binary())
        .args(["bench", "--benchmark", "layer", "--json"])
        .output()
        .expect("Failed to run ny bench");

    assert!(
        output.status.success(),
        "ny bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Linear IBP"),
        "Expected Linear IBP benchmark"
    );
    assert!(stdout.contains("GELU IBP"), "Expected GELU IBP benchmark");
    assert!(
        stdout.contains("LayerNorm IBP"),
        "Expected LayerNorm IBP benchmark"
    );
}

/// Test output bounds are sound (lower <= upper)
#[ntest::timeout(10000)]
#[test]
fn test_bounds_soundness() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.1",
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON and check bounds
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    if let Some(bounds) = v.get("output_bounds").and_then(|b| b.as_array()) {
        for bound in bounds {
            let lower = bound.get("lower").and_then(|l| l.as_f64()).unwrap_or(0.0);
            let upper = bound.get("upper").and_then(|u| u.as_f64()).unwrap_or(0.0);
            assert!(
                lower <= upper,
                "Bound soundness violated: lower {} > upper {}",
                lower,
                upper
            );
        }
    }
}

/// Test that layer-by-layer mode fails on ONNX models without --native
#[ntest::timeout(10000)]
#[test]
fn test_layer_by_layer_requires_native_model() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--layer-by-layer",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        !output.status.success(),
        "Expected failure for --layer-by-layer on ONNX without --native"
    );

    // #395: stderr must be empty when --json emits errors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "Expected empty stderr when --json emits error, got: {}",
        stderr
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value =
        serde_json::from_str(&stdout).expect("Expected valid JSON error output on stdout");
    assert_eq!(
        v.get("error").and_then(|e| e.as_str()),
        Some("unsupported_model_format"),
        "Expected error type 'unsupported_model_format', got: {}",
        stdout
    );
    assert!(
        v.get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .contains("Layer-by-layer"),
        "Expected message to mention layer-by-layer, got: {}",
        stdout
    );
}

/// Test that block-wise mode fails on ONNX models without --native
#[ntest::timeout(10000)]
#[test]
fn test_block_wise_requires_native_model() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--block-wise",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        !output.status.success(),
        "Expected failure for --block-wise on ONNX without --native"
    );

    // #395: stderr must be empty when --json emits errors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "Expected empty stderr when --json emits error, got: {}",
        stderr
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value =
        serde_json::from_str(&stdout).expect("Expected valid JSON error output on stdout");
    assert_eq!(
        v.get("error").and_then(|e| e.as_str()),
        Some("unsupported_model_format"),
        "Expected error type 'unsupported_model_format', got: {}",
        stdout
    );
    assert!(
        v.get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .contains("Block-wise"),
        "Expected message to mention block-wise, got: {}",
        stdout
    );
}

// =============================================================================
// Soundness Provenance Tests
// =============================================================================

/// Test that JSON output includes soundness provenance field
#[ntest::timeout(10000)]
#[test]
fn test_json_output_includes_soundness_provenance() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    // Check that soundness field exists
    assert!(
        v.get("soundness").is_some(),
        "Expected 'soundness' field in JSON output, got: {}",
        stdout
    );

    // Check soundness has mode field
    let soundness = v.get("soundness").unwrap();
    assert!(
        soundness.get("mode").is_some(),
        "Expected 'mode' field in soundness, got: {:?}",
        soundness
    );
}

/// Test that sound mode is reported for basic verification (no heuristics)
#[ntest::timeout(10000)]
#[test]
fn test_soundness_mode_sound_for_basic_verification() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "ibp",
            "--json",
            // Use conservative-layernorm to ensure no heuristics are triggered
            "--conservative-layernorm",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    let soundness = v.get("soundness").expect("Missing soundness field");
    let mode = soundness
        .get("mode")
        .and_then(|m| m.as_str())
        .expect("Missing mode field");

    assert_eq!(
        mode, "sound",
        "Expected 'sound' mode for conservative verification, got: {}",
        mode
    );

    // Should have empty heuristics_used array
    let heuristics = soundness.get("heuristics_used");
    if let Some(h) = heuristics {
        let arr = h.as_array().expect("heuristics_used should be array");
        assert!(
            arr.is_empty(),
            "Expected empty heuristics_used for conservative mode, got: {:?}",
            arr
        );
    }
}

/// Test that CROWN method also includes soundness provenance
#[ntest::timeout(10000)]
#[test]
fn test_soundness_provenance_crown_method() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "crown",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "ny verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    // Check soundness field exists for CROWN method too
    let soundness = v.get("soundness").expect("Missing soundness field");
    let mode = soundness
        .get("mode")
        .and_then(|m| m.as_str())
        .expect("Missing mode field");

    // simple_mlp.onnx doesn't have LayerNorm, so should be sound
    assert!(
        mode == "sound" || mode == "heuristic",
        "Expected valid mode, got: {}",
        mode
    );
}

// =============================================================================
// Require-Sound Gate Tests (#394)
// =============================================================================

/// Test that --require-sound succeeds when verification is sound
#[ntest::timeout(10000)]
#[test]
fn test_require_sound_succeeds_for_sound_verification() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "ibp",
            "--require-sound",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    assert!(
        output.status.success(),
        "Expected success for sound IBP verification with --require-sound: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    // Verify soundness mode is "sound"
    let soundness = v.get("soundness").expect("Missing soundness field");
    let mode = soundness
        .get("mode")
        .and_then(|m| m.as_str())
        .expect("Missing mode field");
    assert_eq!(mode, "sound", "Expected sound mode, got: {}", mode);
}

/// Test that --require-sound forces sound relaxations for heuristic layers
#[ntest::timeout(10000)]
#[test]
fn test_require_sound_forces_sound_relaxations_for_softmax() {
    // softmax.onnx uses Softmax which triggers sampling-based relaxations for CROWN
    let model_path = test_models_dir().join("softmax.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--method",
            "crown",
            "--require-sound",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    // Should succeed with sound relaxations enforced
    assert!(
        output.status.success(),
        "Expected success for sound verification with --require-sound: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify JSON success schema on stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value =
        serde_json::from_str(&stdout).expect("Expected valid JSON output on stdout");

    // #395: Verify stderr is empty when --json mode emits output
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "Expected empty stderr when --json emits output, got: {}",
        stderr
    );

    // Check soundness field exists and has sound mode
    let soundness = v
        .get("soundness")
        .expect("Expected 'soundness' field in output");
    let mode = soundness
        .get("mode")
        .and_then(|m| m.as_str())
        .expect("Expected 'mode' in soundness");
    assert_eq!(mode, "sound", "Expected sound mode, got: {}", mode);

    // heuristics_used should be empty or omitted in sound mode
    let heuristics = soundness.get("heuristics_used");
    assert!(
        heuristics.is_none()
            || heuristics
                .and_then(|h| h.as_array())
                .map(|arr| arr.is_empty())
                .unwrap_or(false),
        "Expected heuristics_used to be empty or omitted for sound mode, got: {:?}",
        heuristics
    );

    // Check status field exists
    let verification_status = v
        .get("status")
        .and_then(|s| s.as_str())
        .expect("Expected 'status' field in output");
    assert!(
        ["verified", "violated", "unknown", "timeout"].contains(&verification_status),
        "Expected valid status, got: {}",
        verification_status
    );
}

/// Test that --require-sound fails for layer-by-layer mode (incompatible)
#[ntest::timeout(10000)]
#[test]
fn test_require_sound_incompatible_with_layer_by_layer() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--layer-by-layer",
            "--require-sound",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    // Should fail
    assert!(
        !output.status.success(),
        "Expected failure for --layer-by-layer with --require-sound"
    );

    // Verify JSON error schema on stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value =
        serde_json::from_str(&stdout).expect("Expected valid JSON error output on stdout");

    // #395: Verify stderr is empty when --json mode emits errors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "Expected empty stderr when --json emits error, got: {}",
        stderr
    );

    // Check error field indicates incompatible options
    let error = v
        .get("error")
        .and_then(|e| e.as_str())
        .expect("Expected 'error' field in JSON output");
    assert_eq!(
        error, "incompatible_options",
        "Expected error type 'incompatible_options', got: {}",
        error
    );

    // Check message mentions layer-by-layer
    let message = v
        .get("message")
        .and_then(|m| m.as_str())
        .expect("Expected 'message' field in JSON output");
    assert!(
        message.contains("layer-by-layer"),
        "Expected message to mention layer-by-layer, got: {}",
        message
    );
}

/// Test that --require-sound fails for block-wise mode (incompatible)
#[ntest::timeout(10000)]
#[test]
fn test_require_sound_incompatible_with_block_wise() {
    let model_path = test_models_dir().join("simple_mlp.onnx");
    require_model(&model_path);

    let output = Command::new(ny_binary())
        .args([
            "verify",
            model_path.to_str().unwrap(),
            "--epsilon",
            "0.01",
            "--block-wise",
            "--require-sound",
            "--json",
        ])
        .output()
        .expect("Failed to run ny verify");

    // Should fail
    assert!(
        !output.status.success(),
        "Expected failure for --block-wise with --require-sound"
    );

    // Verify JSON error schema on stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value =
        serde_json::from_str(&stdout).expect("Expected valid JSON error output on stdout");

    // #395: Verify stderr is empty when --json mode emits errors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.is_empty(),
        "Expected empty stderr when --json emits error, got: {}",
        stderr
    );

    // Check error field indicates incompatible options
    let error = v
        .get("error")
        .and_then(|e| e.as_str())
        .expect("Expected 'error' field in JSON output");
    assert_eq!(
        error, "incompatible_options",
        "Expected error type 'incompatible_options', got: {}",
        error
    );

    // Check message mentions block-wise
    let message = v
        .get("message")
        .and_then(|m| m.as_str())
        .expect("Expected 'message' field in JSON output");
    assert!(
        message.contains("block-wise"),
        "Expected message to mention block-wise, got: {}",
        message
    );
}
