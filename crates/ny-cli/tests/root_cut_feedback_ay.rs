// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual pinned AY V2 hints replayed by the private original-input producer.
//! These authored Source controls do not claim scored-route admission or speed.
#![cfg(feature = "mip")]

use std::io::{Seek, SeekFrom, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, HeadFunctionalLpRequest, FUNCTIONAL_MAX_RESPONSE_BYTES,
    FUNCTIONAL_PROTOCOL_VERSION,
};
use ny_core::head_phase_advice::{HeadPhaseAdvice, HeadPhaseLpHint, HeadPhaseLpRequest};
use ny_propagate::margin_row::{
    register_head_phase_advice_factory, verify_input_head, HeadGroupPolicy, InputHeadOutcome,
    InputHeadReport, InputHeadSpec,
};

#[derive(Default)]
struct ActualAy {
    responses: AtomicUsize,
    signed_responses: AtomicUsize,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadFunctionalLpHint,
}

impl HeadPhaseAdvice for ActualAy {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }

    fn supports_functional_advice(&self) -> bool {
        true
    }

    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        if !request.validate() || Instant::now() >= deadline {
            return None;
        }
        // Regular files prevent pipe backpressure from hiding a stalled child.
        // This test transport exercises the actual helper; production transport
        // identity, pipe limits and child admission have their own provider tests.
        let mut input = tempfile::tempfile().ok()?;
        serde_json::to_writer(&mut input, request).ok()?;
        input.flush().ok()?;
        input.seek(SeekFrom::Start(0)).ok()?;
        let output = tempfile::NamedTempFile::new().ok()?;
        let errors = tempfile::tempfile().ok()?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
        let mut child = command
            .arg("__head-functional-ay-advice")
            .stdin(Stdio::from(input))
            .stdout(Stdio::from(output.reopen().ok()?))
            .stderr(Stdio::from(errors))
            .spawn()
            .ok()?;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
            }
        };
        if !status.success()
            || Instant::now() >= deadline
            || output.as_file().metadata().ok()?.len() > FUNCTIONAL_MAX_RESPONSE_BYTES as u64
        {
            return None;
        }
        let response: Response = serde_json::from_reader(output.reopen().ok()?).ok()?;
        if response.protocol_version != FUNCTIONAL_PROTOCOL_VERSION
            || response.ay_revision != "0fb75da29369a25e9c5d6a296f299eec2bb61f35"
            || !response.hint.validate(request)
            || Instant::now() >= deadline
        {
            return None;
        }
        self.responses.fetch_add(1, Ordering::SeqCst);
        let signed = request.feature_count == 65
            && request.bounds.len() == 138
            && request.rows.iter().any(|row| {
                let positive = row
                    .coefficients
                    .iter()
                    .filter(|&&(j, a)| j >= 65 && a > 0.0)
                    .count();
                let negative = row
                    .coefficients
                    .iter()
                    .filter(|&&(j, a)| j >= 65 && a < 0.0)
                    .count();
                positive == 65 && negative == 8
            });
        if signed {
            self.signed_responses.fetch_add(1, Ordering::SeqCst);
        }
        Some(response.hint)
    }
}

fn cycles(bias: f32, permuted: bool) -> InputHeadSpec {
    let mut first = vec![0.0; 35 * 35];
    let input = |i: usize| if permuted { (12 * i + 11) % 35 } else { i };
    let head = |b: usize, j: usize| {
        if permuted {
            ((3 * b + 2) % 5) * 7 + (3 * j + 1) % 7
        } else {
            7 * b + j
        }
    };
    for b in 0..5 {
        for j in 0..7 {
            let h = head(b, j);
            first[input(7 * b + j) * 35 + h] = 1.0;
            first[input(7 * b + (j + 1) % 7) * 35 + h] = -1.0;
        }
    }
    let weights = (0..35).flat_map(|_| [-1.0, 0.0]).collect();
    InputHeadSpec::new(
        35,
        35,
        first,
        vec![0.0; 35],
        weights,
        vec![bias, 0.0],
        [71; 32],
    )
    .unwrap()
}

fn affine(bias: f32) -> InputHeadSpec {
    let mut first = vec![0.0; 2 * 16];
    for j in 0..14 {
        first[j] = 1.0;
        first[16 + j] = 1.0;
    }
    first[14] = 1.0; // p = u+2 is stable positive.
    first[16 + 15] = -1.0; // n = 2-v is stable positive.
    let mut first_bias = vec![0.0; 16];
    first_bias[14..].fill(2.0);
    let weights = (0..16)
        .flat_map(|j| [if j < 14 { -1.0 } else { -14.0 }, 0.0])
        .collect();
    InputHeadSpec::new(2, 16, first, first_bias, weights, vec![bias, 0.0], [72; 32]).unwrap()
}

fn run(spec: &InputHeadSpec, safe: bool, provider: &ActualAy) -> InputHeadReport {
    let responses_before = provider.responses.load(Ordering::SeqCst);
    let outcome = verify_input_head(
        spec,
        &vec![-1.0; spec.inputs()],
        &vec![1.0; spec.inputs()],
        0,
        1,
        HeadGroupPolicy::new(5.0, 2, 4).unwrap(),
        Instant::now() + Duration::from_secs(8),
    );
    let report = match outcome {
        InputHeadOutcome::Unsat(report, _) if safe => report,
        InputHeadOutcome::Unknown(report) if !safe => report,
        InputHeadOutcome::Unknown(report) => panic!("safe Source control stayed open: {report:?}"),
        InputHeadOutcome::Unsat(report, _) => panic!("unsafe Source control certified: {report:?}"),
    };
    let lower = report.lower.expect("non-VOID Source replay");
    if safe {
        assert!(lower > 0.49 && lower <= 0.5, "{report:?}");
    } else {
        assert!(lower <= -0.5, "{report:?}");
    }
    assert!(report.directional_cuts > 0, "{report:?}");
    if !safe {
        assert!(report.directional_ay_replayed > 0, "{report:?}");
    }
    assert!(
        provider.responses.load(Ordering::SeqCst) > responses_before,
        "AY never responded: {report:?}"
    );
    assert!(report.directional_masks <= 8192, "{report:?}");
    assert!(report.directional_work_cells <= 2_000_000, "{report:?}");
    report
}

fn signed_dense(bias: f32) -> InputHeadSpec {
    let (epsilon, delta) = (1.0f32 / 4096.0, 1.0f32 / 256.0);
    let mut first = vec![0.0; 65 * 73];
    for i in 0..65 {
        for j in 0..65 {
            first[i * 73 + j] = epsilon + if i == j { 1.0 } else { 0.0 };
        }
        for k in 0..8 {
            first[i * 73 + 65 + k] =
                1.0 + 65.0 * epsilon + delta * epsilon + if i == k { delta } else { 0.0 };
        }
    }
    let weights = (0..73)
        .flat_map(|j| {
            [
                if j < 8 {
                    8.0 + delta
                } else if j < 65 {
                    8.0
                } else {
                    -1.0
                },
                0.0,
            ]
        })
        .collect();
    InputHeadSpec::new(
        65,
        73,
        first,
        vec![0.0; 73],
        weights,
        vec![bias, 0.0],
        [73; 32],
    )
    .unwrap()
}

fn run_signed(safe: bool, provider: &ActualAy) {
    let before = provider.signed_responses.load(Ordering::SeqCst);
    let spec = signed_dense(if safe { 0.5 } else { -0.5 });
    let outcome = verify_input_head(
        &spec,
        &[-1.0; 65],
        &[1.0; 65],
        0,
        1,
        HeadGroupPolicy::new(5.0, 2, 4).unwrap(),
        Instant::now() + Duration::from_secs(8),
    );
    let report = match outcome {
        InputHeadOutcome::Unsat(report, _) if safe => report,
        InputHeadOutcome::Unknown(report) if !safe => report,
        InputHeadOutcome::Unknown(report) => {
            panic!("safe dense signed control stayed open: {report:?}")
        }
        InputHeadOutcome::Unsat(report, _) => {
            panic!("unsafe dense signed control certified: {report:?}")
        }
    };
    let floor = report.lower.expect("completed original-margin replay");
    assert!(
        if safe {
            floor > 0.45 && floor <= 0.5
        } else {
            floor <= -0.5
        },
        "{report:?}"
    );
    assert!(
        report.directional_cuts >= 1 && report.directional_native_replayed >= 1,
        "{report:?}"
    );
    assert!(
        report.directional_masks <= 8192 && report.directional_work_cells <= 2_000_000,
        "{report:?}"
    );
    assert!(
        report.directional_rounds <= 3 && report.directional_advice_calls <= 3,
        "{report:?}"
    );
    if !safe {
        assert!(
            provider.signed_responses.load(Ordering::SeqCst) > before,
            "no actual well-formed signed AY response: {report:?}"
        );
        assert!(
            report.directional_ay_replayed > 0,
            "signed AY response was not Source replayed: {report:?}"
        );
    }
}

#[test]
fn actual_ay_feedback_and_input_affine_source_controls() {
    // A single test owns the process-global factory and serializes its helpers.
    let provider = Arc::new(ActualAy::default());
    let factory = Arc::clone(&provider);
    register_head_phase_advice_factory(move || Some(factory.clone()));

    for permuted in [false, true] {
        let report = run(&cycles(30.5, permuted), true, &provider);
        assert!(report.directional_rounds >= 2, "{report:?}");
        assert!(report.directional_cuts >= 5, "{report:?}");
    }
    let report = run(&cycles(29.5, false), false, &provider);
    assert!(report.directional_advice_calls >= 2, "{report:?}");
    for (bias, safe) in [(84.5, true), (83.5, false)] {
        let report = run(&affine(bias), safe, &provider);
        assert!(report.directional_affine_cuts >= 2, "{report:?}");
    }
    assert!(provider.responses.load(Ordering::SeqCst) >= 5);
    // Authored65-input Source/V2 integration, not scalarRealV1 scored CLI.
    // Legacy five controls retain their existing >.49 and actual-AY assertions.
    run_signed(true, &provider);
    run_signed(false, &provider);
}
