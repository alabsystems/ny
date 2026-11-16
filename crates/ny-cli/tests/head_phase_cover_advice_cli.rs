// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real V3 hidden transport; no ordinary model/device/logger initialization.
#![cfg(feature = "mip")]

use ny_core::head_functional_advice::HeadFunctionalLpRow;
use ny_core::head_phase_cover_advice::{
    PhaseCoverLeaf, PhaseCoverLpHint, PhaseCoverLpRequest, PHASE_COVER_MAX_RESPONSE_BYTES,
    PHASE_COVER_PROTOCOL_VERSION,
};
use std::io::{Seek, SeekFrom, Write};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const HELPER: &str = "__head-phase-cover-ay-advice";

fn helper(args: &[&str], payload: Option<&[u8]>) -> Output {
    let stdin = if let Some(bytes) = payload {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        Stdio::from(file)
    } else {
        Stdio::null()
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_ny"))
        .args(args)
        .env("NY_DISABLE_CUDA", "not-a-boolean")
        .env("NY_WGPU_CROWN", "not-a-policy")
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "phase-cover helper exceeded watchdog: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

fn clean_error(output: Output) {
    assert_eq!(output.status.code(), Some(4));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error:"));
    assert!(!stderr.contains("NY_DISABLE_CUDA must") && !stderr.contains("NY_WGPU_CROWN must"));
}

fn request() -> PhaseCoverLpRequest {
    PhaseCoverLpRequest {
        protocol_version: PHASE_COVER_PROTOCOL_VERSION,
        selected_heads: 1,
        bounds: vec![[0.0, 1.0]],
        objective: vec![1.0],
        rows: Vec::new(),
        leaves: vec![
            PhaseCoverLeaf {
                mask: 0,
                rows: vec![HeadFunctionalLpRow {
                    coefficients: vec![(0, -1.0)],
                    lower: 0.0,
                }],
            },
            PhaseCoverLeaf {
                mask: 1,
                rows: vec![HeadFunctionalLpRow {
                    coefficients: vec![(0, 1.0)],
                    lower: 0.5,
                }],
            },
        ],
    }
}

#[test]
fn phase_cover_rejects_malformed_and_argument_payload_before_device_startup() {
    clean_error(helper(&[HELPER], Some(b"{}")));
    clean_error(helper(&["-vv", HELPER], Some(b"{}")));
    clean_error(helper(&[HELPER, "{}"], None));
    let valid = request();
    for remaining in [0u64, 750_001] {
        let payload = serde_json::to_vec(
            &serde_json::json!({"remaining_micros": remaining, "request": valid}),
        )
        .unwrap();
        clean_error(helper(&[HELPER], Some(&payload)));
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    protocol_version: u32,
    ay_revision: String,
    hint: PhaseCoverLpHint,
}

#[test]
fn actual_phase_cover_hidden_ay_returns_only_replayable_pinned_multipliers() {
    let request = request();
    let payload =
        serde_json::to_vec(&serde_json::json!({"remaining_micros": 700_000, "request": request}))
            .unwrap();
    let output = helper(&[HELPER], Some(&payload));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= PHASE_COVER_MAX_RESPONSE_BYTES);
    let response: Response = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response.protocol_version, PHASE_COVER_PROTOCOL_VERSION);
    assert_eq!(response.ay_revision, ny_mip::head_phase_ay::ay_revision());
    assert!(response.hint.validate(&request));
    assert_eq!(response.hint.leaves.len(), 2);
    let hint = response.hint.leaves.iter().find(|h| h.mask == 1).unwrap();
    let best = [1.0f64, -1.0]
        .into_iter()
        .map(|sign| {
            let dual = (sign * hint.row_duals[0]).max(0.0);
            0.5 * dual + (1.0 - dual).min(0.0)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(best > 0.25 && best <= 0.5);
}
