// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit launcher for external-data and hardware conformance suites.
//!
//! These suites are deliberately not ordinary unit tests: their inputs are
//! large generated exports, separately downloaded VNN-COMP corpora, or a live
//! WGPU proof device.  Selecting a suite registers all of its tests, and every
//! selected test fails if its own required input is unavailable.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, ExitCode};

const USAGE: &str = "\
Usage: ny_onnx_conformance <suite> [cargo-test arguments...]

Suites:
  avoice       Real AVoice exported-model tests
  avoice-gpu   AVoice exported-model tests plus live WGPU tests
  whisper      Generated Whisper encoder tests
  vnncomp      Downloaded VNN-COMP model/property tests
  vnncomp-gpu  VNN-COMP tests plus live WGPU tests
  wgpu         Hermetic-model tests requiring a live WGPU device
  all          Every external-data and hardware test

Examples:
  cargo run -p ny-onnx --bin ny_onnx_conformance -- whisper --lib
  cargo run -p ny-onnx --bin ny_onnx_conformance -- vnncomp --release
  cargo run -p ny-onnx --release --features external-vnncomp \
      --example vit-zero-width-node-bisect -- scan

Model-backed tests honor NY_TEST_MODELS_DIR.  Individual tests print their
exact fixture-generation/download requirement and fail when it is absent.
Builds are serialized by default to reduce peak memory; an existing
CARGO_BUILD_JOBS setting is preserved.
";

fn suite_features(suite: &str) -> Option<&'static str> {
    match suite {
        "avoice" => Some("external-avoice"),
        "avoice-gpu" => Some("external-avoice,external-wgpu"),
        "whisper" => Some("external-whisper"),
        "vnncomp" => Some("external-vnncomp"),
        "vnncomp-gpu" => Some("external-vnncomp,external-wgpu"),
        "wgpu" => Some("external-wgpu"),
        "all" => Some("external-conformance"),
        _ => None,
    }
}

fn set_default_env(command: &mut Command, key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        command.env(key, value);
    }
}

fn quoted(argument: &OsStr) -> String {
    let rendered = argument.to_string_lossy();
    if rendered
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_=.,/:".contains(ch))
    {
        rendered.into_owned()
    } else {
        format!("{rendered:?}")
    }
}

fn main() -> ExitCode {
    let mut arguments = std::env::args_os().skip(1);
    let Some(suite_os) = arguments.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let Some(suite) = suite_os.to_str() else {
        eprintln!("suite name must be valid UTF-8\n\n{USAGE}");
        return ExitCode::from(2);
    };
    if matches!(suite, "-h" | "--help" | "help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let Some(features) = suite_features(suite) else {
        eprintln!("unknown conformance suite {suite:?}\n\n{USAGE}");
        return ExitCode::from(2);
    };

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut command = Command::new(cargo);
    command
        .current_dir(workspace)
        .args(["test", "-p", "ny-onnx", "--features", features]);
    command.args(arguments);
    set_default_env(&mut command, "CARGO_BUILD_JOBS", "1");
    set_default_env(&mut command, "RAYON_NUM_THREADS", "1");
    set_default_env(&mut command, "TOKIO_WORKER_THREADS", "1");
    set_default_env(&mut command, "MALLOC_ARENA_MAX", "2");

    eprintln!(
        "running external {suite} conformance lane:\n  {}",
        std::iter::once(command.get_program())
            .chain(command.get_args())
            .map(quoted)
            .collect::<Vec<_>>()
            .join(" ")
    );
    match command.status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(1, 255) as u8),
        Err(error) => {
            eprintln!("failed to launch cargo test: {error}");
            ExitCode::FAILURE
        }
    }
}
