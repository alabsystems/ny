// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored Source/V2 multirow controls; these are not scored CLI instances.
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
    InputHeadSpec,
};

#[derive(Default)]
struct ActualAy {
    joint_responses: AtomicUsize,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadFunctionalLpHint,
}

fn has_two_disjoint_signed_rows(request: &HeadFunctionalLpRequest) -> bool {
    let mut supports = Vec::<Vec<usize>>::new();
    for row in &request.rows {
        let mut positive = false;
        let mut negative = Vec::new();
        for &(j, a) in &row.coefficients {
            if j >= request.feature_count {
                positive |= a > 0.0;
                if a < 0.0 {
                    negative.push(j);
                }
            }
        }
        // Scalar chords and old positive-cone rows have at most one negative
        // head. Require two independently supported mixed-sign rows together.
        if positive && negative.len() >= 2 {
            if supports
                .iter()
                .any(|old| old.iter().all(|j| !negative.contains(j)))
            {
                return true;
            }
            supports.push(negative);
        }
    }
    false
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
        let joint = has_two_disjoint_signed_rows(request);
        // Regular files keep child pipe capacity out of this transport control.
        let mut input = tempfile::tempfile().ok()?;
        serde_json::to_writer(&mut input, request).ok()?;
        input.flush().ok()?;
        input.seek(SeekFrom::Start(0)).ok()?;
        let output = tempfile::NamedTempFile::new().ok()?;
        let errors = tempfile::tempfile().ok()?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_ny"))
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
        if joint {
            self.joint_responses.fetch_add(1, Ordering::SeqCst);
        }
        Some(response.hint)
    }
}

fn dense_two_blocks(bias: f32, permuted: bool) -> InputHeadSpec {
    let (d, h) = (80, 96);
    let (epsilon, delta) = (1.0f32 / 4096.0, 1.0f32 / 256.0);
    let input = |i: usize| if permuted { (11 * i + 9) % d } else { i };
    let head = |j: usize| if permuted { (17 * j + 13) % h } else { j };
    let mut first = vec![0.0; d * h];
    let mut output = vec![0.0; h * 2];
    for i in 0..d {
        for p in 0..d {
            first[input(i) * h + head(p)] = epsilon + if i == p { 1.0 } else { 0.0 };
        }
        for group in 0..2 {
            for k in 0..8 {
                first[input(i) * h + head(d + 8 * group + k)] = (40.0 + delta) * epsilon
                    + if i / 40 == group { 1.0 } else { 0.0 }
                    + if i == group * 40 + k { delta } else { 0.0 };
            }
        }
    }
    assert!(first.iter().all(|v| v.is_finite() && *v > 0.0));
    for p in 0..d {
        output[2 * head(p)] = 8.0 + if p % 40 < 8 { delta } else { 0.0 };
    }
    for n in d..h {
        output[2 * head(n)] = -1.0;
    }
    // Every first bias is zero: the original x=0 counterexample evaluates
    // every hidden head to zero, leaving exactly the unsafe sibling's -0.5.
    InputHeadSpec::new(d, h, first, vec![0.0; h], output, vec![bias, 0.0], [84; 32]).unwrap()
}

#[test]
fn actual_ay_replays_two_allocated_signed_rows_in_one_original_objective() {
    // This one test owns its process-global factory and serializes all helpers.
    let provider = Arc::new(ActualAy::default());
    let factory = Arc::clone(&provider);
    register_head_phase_advice_factory(move || Some(factory.clone()));
    for permuted in [false, true] {
        for safe in [true, false] {
            let before = provider.joint_responses.load(Ordering::SeqCst);
            let spec = dense_two_blocks(if safe { 0.5 } else { -0.5 }, permuted);
            let outcome = verify_input_head(
                &spec,
                &[-1.0; 80],
                &[1.0; 80],
                0,
                1,
                HeadGroupPolicy::new(5.0, 2, 4).unwrap(),
                Instant::now() + Duration::from_secs(8),
            );
            let report = match outcome {
                InputHeadOutcome::Unsat(report, _) if safe => report,
                InputHeadOutcome::Unknown(report) if !safe => report,
                InputHeadOutcome::Unknown(report) => {
                    panic!("safe allocated Source control stayed open: {report:?}")
                }
                InputHeadOutcome::Unsat(report, _) => {
                    panic!("original unsafe x=0 was certified: {report:?}")
                }
            };
            let lower = report.lower.expect("non-VOID original-objective replay");
            assert!(
                if safe {
                    lower > 0.4 && lower <= 0.5
                } else {
                    lower <= -0.5
                },
                "{report:?}"
            );
            assert!(report.directional_cuts >= 2, "{report:?}");
            assert!(report.directional_native_replayed > 0, "{report:?}");
            assert!(report.directional_masks <= 8192, "{report:?}");
            assert!(report.directional_work_cells <= 2_000_000, "{report:?}");
            assert!(report.directional_attempted <= 32, "{report:?}");
            assert!(report.directional_rounds <= 3, "{report:?}");
            assert!(report.directional_advice_calls <= 3, "{report:?}");
            if !safe {
                assert!(report.directional_ay_replayed > 0, "{report:?}");
                assert!(
                    provider.joint_responses.load(Ordering::SeqCst) > before,
                    "no actual AY response to two mixed rows in one V2 model: {report:?}"
                );
            }
        }
    }
}
