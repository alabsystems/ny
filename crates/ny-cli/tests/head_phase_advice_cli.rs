// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The private AY transport bypasses logger/backend bootstrap. Malformed
//! requests decline before any solve. The real wire fixture checks actual AY
//! hints by exact dyadic source replay, without trusting an objective/status.
#![cfg(feature = "mip")]

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use ny_core::head_phase_advice::{
    HeadPhaseLpHint, HeadPhaseLpRequest, HeadPhaseLpRow, MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
};

fn helper(args: &[&str], request: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ny"))
        .args(args)
        // Normal bootstrap rejects these exact boolean policies. The helper
        // must not read them or initialize a device even if its JSON is bad.
        .env("NY_DISABLE_CUDA", "not-a-boolean")
        .env("NY_WGPU_CROWN", "not-a-policy")
        .stdin(if request.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn private advice helper");
    if let Some(request) = request {
        child.stdin.take().unwrap().write_all(request).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().expect("poll helper").is_none() {
        if Instant::now() >= deadline {
            // The child may exit between the poll and kill; always reap it.
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap stalled test helper");
            panic!(
                "helper exceeded test deadline; stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().expect("collect helper exit")
}

fn assert_clean_error(output: Output) {
    assert_eq!(output.status.code(), Some(4));
    assert!(
        output.stdout.is_empty(),
        "errors must not contaminate hint stdout"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error:"));
    assert!(!stderr.contains("NY_DISABLE_CUDA must"));
    assert!(!stderr.contains("NY_WGPU_CROWN must"));
}

#[test]
fn malformed_helper_json_fails_before_device_or_logger_setup() {
    assert_clean_error(helper(&["__head-phase-ay-advice"], Some(b"{}")));
    // The Clap route with global verbosity must share the early return.
    assert_clean_error(helper(&["-vv", "__head-phase-ay-advice"], Some(b"{}")));
}

#[test]
fn helper_rejects_argument_supplied_payload_without_reading_stdin() {
    assert_clean_error(helper(&["__head-phase-ay-advice", "{}"], None));
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResponse {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadPhaseLpHint,
}

fn ratio(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

#[test]
fn actual_ay_helper_duals_pass_exact_shifted_source_phase_replay() {
    // These rows represent the authored dyadic y1=x, y2=x+1/2 source,
    // with explicit active-phase rows. The test never reads an AY optimum,
    // status, primal model or claimed neural bound.
    let request = HeadPhaseLpRequest {
        protocol_version: PROTOCOL_VERSION,
        bounds: vec![[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.5]],
        objective: vec![0.0, 1.0, -3.0 / 8.0],
        rows: vec![
            HeadPhaseLpRow {
                coefficients: vec![-1.0, 1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, -1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![-1.0, 0.0, 1.0],
                lower: 0.5,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, 0.0, -1.0],
                lower: -0.5,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 0.0, 1.0],
                lower: 0.0,
            },
        ],
        phase_rows: vec![2, 5],
    };
    assert!(request.validate());
    let encoded = serde_json::to_vec(&request).unwrap();
    let output = helper(&["__head-phase-ay-advice"], Some(&encoded));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= MAX_MESSAGE_BYTES);
    let response: WireResponse = serde_json::from_slice(&output.stdout)
        .expect("stdout must contain only the strict bounded hint envelope");
    assert_eq!(response.protocol_version, PROTOCOL_VERSION);
    assert_eq!(
        response.ay_revision,
        "0fb75da29369a25e9c5d6a296f299eec2bb61f35"
    );
    assert!(response.hint.validate(&request));

    // Binary64 values become exact rationals. Both sign conventions offer
    // finite nonnegative p; source weak-dual replay alone decides usefulness.
    let mut replayed_positive = false;
    for sign in [1.0, -1.0] {
        let a = BigRational::from_float((sign * response.hint.row_duals[2]).max(0.0)).unwrap();
        let b = BigRational::from_float((sign * response.hint.row_duals[5]).max(0.0)).unwrap();
        let lower = ratio(13, 64) - (ratio(3, 8) + &b) / ratio(2, 1) - (ratio(5, 8) - a - b).abs();
        assert!(
            lower <= ratio(1, 64),
            "phase replay exceeded its exact source boundary value"
        );
        replayed_positive |= lower > BigRational::zero();
    }
    assert!(
        replayed_positive,
        "actual AY hints failed independent exact source replay"
    );
    // This checks one active phase and real transport. Complete private
    // source-cover/token acceptance is separately tested in ny-propagate.
}
