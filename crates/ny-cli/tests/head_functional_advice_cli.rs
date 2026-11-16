// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual V2 helper transport plus exact authored scalar-functional replay.
//! These checks do not import AY status, objective, primal or neural authority.
#![cfg(feature = "mip")]

use std::io::{Seek, SeekFrom, Write};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, HeadFunctionalLpRequest, HeadFunctionalLpRow,
    FUNCTIONAL_MAX_RESPONSE_BYTES, FUNCTIONAL_PROTOCOL_VERSION,
};

const HELPER: &str = "__head-functional-ay-advice";

fn helper(args: &[&str], request: Option<&[u8]>) -> Output {
    // A regular stdin file keeps the test's request write from waiting on a
    // broken helper before its watchdog can start. Production bounded pipe
    // transport and cleanup remain separate ny-mip tests.
    let stdin = match request {
        Some(bytes) => {
            let mut file = tempfile::tempfile().expect("temporary request stdin");
            file.write_all(bytes).unwrap();
            file.seek(SeekFrom::Start(0)).unwrap();
            Stdio::from(file)
        }
        None => Stdio::null(),
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_ny"))
        .args(args)
        .env("NY_DISABLE_CUDA", "not-a-boolean")
        .env("NY_WGPU_CROWN", "not-a-policy")
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn private global-functional helper");
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().expect("poll helper").is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap stalled helper");
            panic!(
                "functional helper exceeded test watchdog; stderr: {}",
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
        "failure must leave hint stdout empty"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error:"));
    assert!(!stderr.contains("NY_DISABLE_CUDA must"));
    assert!(!stderr.contains("NY_WGPU_CROWN must"));
}

#[test]
fn malformed_functional_helper_json_declines_before_device_or_logger_setup() {
    assert_clean_error(helper(&[HELPER], Some(b"{}")));
    assert_clean_error(helper(&["-vv", HELPER], Some(b"{}")));
}

#[test]
fn functional_helper_rejects_payload_arguments_without_reading_stdin() {
    assert_clean_error(helper(&[HELPER, "{}"], None));
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResponse {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadFunctionalLpHint,
}

fn ratio(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn relu(x: &BigRational) -> BigRational {
    if x > &BigRational::zero() {
        x.clone()
    } else {
        BigRational::zero()
    }
}

fn scalar_intercept(
    q: &BigRational,
    slope: &BigRational,
    lo: BigRational,
    hi: BigRational,
) -> BigRational {
    // This continuous piecewise-affine function reaches its minimum at an
    // endpoint or at the kink. Every unselected scalar intercept is charged.
    let mut values = vec![q * relu(&lo) - slope * &lo, q * relu(&hi) - slope * &hi];
    if lo <= BigRational::zero() && hi >= BigRational::zero() {
        values.push(BigRational::zero());
    }
    values.into_iter().min().unwrap()
}

#[test]
fn actual_functional_ay_helper_slopes_pass_exact_six_dimensional_source_replay() {
    // Features z_i=x_i+2 belong to [1,3]. Epigraphs u_i have both GEQ
    // rows u_i>=0 and u_i-z_i>=-2. Stable y7=sum(z)-5 carries slope -3/8.
    // No selected phase rows: G is empty, so one globally valid functional
    // covers all six unstable head units simultaneously.
    let mut rows = Vec::new();
    for i in 0..6 {
        rows.push(HeadFunctionalLpRow {
            coefficients: vec![(6 + i, 1.0)],
            lower: 0.0,
        });
        rows.push(HeadFunctionalLpRow {
            coefficients: vec![(i, -1.0), (6 + i, 1.0)],
            lower: -2.0,
        });
    }
    let request = HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: 6,
        bounds: [vec![[1.0, 3.0]; 6], vec![[0.0, 1.0]; 6]].concat(),
        objective: [vec![-3.0 / 8.0; 6], vec![1.0; 6]].concat(),
        rows,
        slope_rows: vec![Some(1), Some(3), Some(5), Some(7), Some(9), Some(11), None],
        phase_rows: Vec::new(),
    };
    assert!(request.validate());
    let encoded = serde_json::to_vec(&request).unwrap();
    let output = helper(&[HELPER], Some(&encoded));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= FUNCTIONAL_MAX_RESPONSE_BYTES);
    let response: WireResponse = serde_json::from_slice(&output.stdout)
        .expect("strict V2 hint envelope without stdout contamination");
    assert_eq!(response.protocol_version, FUNCTIONAL_PROTOCOL_VERSION);
    assert_eq!(response.ay_revision, ny_mip::head_phase_ay::ay_revision());
    assert!(response.hint.validate(&request));

    let mut positive = false;
    for sign in [1.0, -1.0] {
        let mut mismatch = BigRational::zero();
        let mut intercept = BigRational::zero();
        for row in request.slope_rows.iter().flatten() {
            let slope =
                BigRational::from_float((sign * response.hint.row_duals[*row]).clamp(0.0, 1.0))
                    .unwrap();
            intercept += scalar_intercept(&ratio(1, 1), &slope, ratio(-1, 1), ratio(1, 1));
            mismatch += (slope - ratio(3, 8)).abs();
        }
        let stable = ratio(-3, 8);
        intercept += scalar_intercept(&stable, &stable, ratio(1, 1), ratio(13, 1));
        assert_eq!(intercept, BigRational::zero());
        // Exact source replay, including all seven scalar intercepts. The
        // original authored dyadic Q has zero discrepancy in this analytic
        // fixture; production NY still prices its full directed Q/Y residual.
        let lower = ratio(43, 16) - ratio(21, 8) - &mismatch + &intercept;
        let negative = ratio(41, 16) - ratio(21, 8) - mismatch + intercept;
        assert!(lower <= ratio(1, 16));
        assert!(negative <= ratio(-1, 16));
        positive |= lower > BigRational::zero();
    }
    assert!(
        positive,
        "actual V2 hints did not improve the authored source functional"
    );
    // Transport + analytic source evidence only. Actual private root-pack,
    // empty-group admission, complete cover and ledger token are separately
    // exercised by the source-producer tests, pending parent validation.
}
