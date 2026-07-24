// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_test_utils::workspace_root;
use std::path::Path;
use std::process::{Command, Output};

pub(crate) fn ny_binary() -> String {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ny") {
        return bin;
    }

    let workspace = workspace_root();
    if let Ok(worker_id) = std::env::var("AI_WORKER_ID") {
        let worker_debug_alias_bin = workspace.join(format!("target/worker_{worker_id}/debug/ny"));
        if worker_debug_alias_bin.exists() {
            return worker_debug_alias_bin.to_string_lossy().to_string();
        }
        let worker_release_alias_bin =
            workspace.join(format!("target/worker_{worker_id}/release/ny"));
        if worker_release_alias_bin.exists() {
            return worker_release_alias_bin.to_string_lossy().to_string();
        }
        let worker_debug_bin = workspace.join(format!("target/worker_{worker_id}/debug/ny"));
        if worker_debug_bin.exists() {
            return worker_debug_bin.to_string_lossy().to_string();
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

#[allow(dead_code)] // Shared integration-test helper; not every test binary uses benchmark assets.
pub(crate) fn require_benchmark_file(path: &Path) {
    assert!(
        path.exists(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );
}

pub(crate) fn run_ny(args: &[&str]) -> Output {
    Command::new(ny_binary())
        .current_dir(workspace_root())
        .args(args)
        .output()
        .expect("failed to execute ny binary")
}

#[allow(dead_code)] // Shared integration-test helper; not every test binary parses JSON output.
pub(crate) fn parse_json_output(output: &Output, command_label: &str) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!(
            "failed to parse JSON output for {command_label}: {e}\nstdout: {stdout}\nstderr: {stderr}"
        )
    })
}
