// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;

fn ny_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ny"))
}

#[test]
fn every_whisper_verdict_command_fails_before_opening_the_model() {
    let missing_model = "/definitely/nonexistent/ny-whisper-model.onnx";

    for command in [
        "whisper",
        "whisper-seq",
        "whisper-sweep",
        "whisper-eps-search",
    ] {
        for json in [false, true] {
            let mut invocation = Command::new(ny_binary());
            invocation.arg(command).arg(missing_model);
            if json {
                invocation.arg("--json");
            }
            let output = invocation
                .output()
                .unwrap_or_else(|error| panic!("failed to run `{command}`: {error}"));

            assert!(
                !output.status.success(),
                "`{command}` unexpectedly returned success (json={json})"
            );
            assert!(
                output.stdout.is_empty(),
                "`{command}` emitted verdict-like stdout (json={json}): {}",
                String::from_utf8_lossy(&output.stdout)
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("Whisper verification is unavailable")
                    && stderr.contains("no verification verdict was produced"),
                "`{command}` returned the wrong fail-closed error (json={json}): {stderr}"
            );
            assert!(
                !stderr.contains(missing_model),
                "`{command}` inspected or reported the missing model before failing closed: \
                 {stderr}"
            );
        }
    }
}
