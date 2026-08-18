// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Launch the exhaustive Snake/comparison/piecewise-constant envelope audit.
//!
//! This is intentionally an explicit executable: the exhaustive profile probes
//! hundreds of millions of points and is inappropriate for an ordinary unit
//! test run. The same tests always execute a bounded deterministic corpus in CI.

use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");

    let status = Command::new(cargo)
        .arg("test")
        .arg("--release")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--lib")
        .arg("envelope_audit_snakecompare::audit_")
        .arg("--")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("NY_ENVELOPE_AUDIT_SNAKECOMPARE", "exhaustive")
        .status()
        .expect("failed to launch the exhaustive envelope audit");

    if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
