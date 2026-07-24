// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
#[cfg(feature = "mip")]
use ny_core::{Bound, VerificationResult};
use ny_propagate::{BabVerificationStatus, BetaCrownResult};
use std::path::{Path, PathBuf};
#[cfg(feature = "mip")]
use std::time::Duration;

use crate::commands::verify::{exit_codes, json_f32};

use std::cell::RefCell;

thread_local! {
    /// In-process capture sink for the competition JSON verdict.
    ///
    /// When `Some`, the `--json` output sites (`output_result`, the SMT path, and
    /// the HiGHS-MIP path) store the rendered competition JSON string here instead
    /// of printing it to stdout, and they SKIP the `std::process::exit(...)` that a
    /// non-verified verdict would normally trigger. This lets the native `vnncomp`
    /// subcommand call `handle_beta_crown_command` directly (no shell-out, no second
    /// `ny` process), capture the exact same verdict JSON the shell wrapper used to
    /// parse, and translate it into the VNN-COMP result string in-process.
    ///
    /// SOUNDNESS: the captured string is byte-for-byte the same JSON the CLI would
    /// have printed in `--json` mode — the verdict mapping is unchanged. Suppressing
    /// the process exit is purely a control-flow concern (the exit code is a
    /// CLI-ergonomics signal, not part of the soundness contract; the VNN-COMP result
    /// is carried entirely by the JSON `status`/`counterexample_vnnlib` fields).
    static CAPTURE_SINK: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Begin capturing the competition JSON verdict on this thread.
///
/// Any previously captured value is cleared. Call [`take_captured_json`] afterwards
/// to retrieve the rendered verdict, and [`end_capture`] (or the returned value's
/// drop) to stop capturing.
pub(crate) fn begin_capture() {
    CAPTURE_SINK.with(|sink| *sink.borrow_mut() = Some(String::new()));
}

/// Stop capturing on this thread, discarding any buffered verdict.
pub(crate) fn end_capture() {
    CAPTURE_SINK.with(|sink| *sink.borrow_mut() = None);
}

/// Take the captured competition JSON verdict, if capture is active and a verdict
/// was rendered. Returns `None` when not capturing or when no verdict was produced.
pub(crate) fn take_captured_json() -> Option<String> {
    CAPTURE_SINK.with(|sink| {
        let mut guard = sink.borrow_mut();
        match guard.as_mut() {
            Some(buf) if !buf.is_empty() => Some(std::mem::take(buf)),
            _ => None,
        }
    })
}

/// Returns `true` when the competition JSON verdict should be captured rather than
/// printed (and the process-exit suppressed). Used by the MIP/SMT `--json` output
/// sites to decide whether to skip `std::process::exit`. Only referenced from the
/// `mip`-gated paths, so it is dead code in a default (non-mip) build.
#[cfg_attr(not(feature = "mip"), allow(dead_code))]
pub(super) fn is_capturing() -> bool {
    CAPTURE_SINK.with(|sink| sink.borrow().is_some())
}

/// Route a rendered competition JSON verdict to the active capture sink (if any) or
/// to stdout. Returns `true` when the verdict was captured (and the caller MUST NOT
/// `std::process::exit`), `false` when it was printed normally.
pub(super) fn emit_competition_json(json: &str) -> bool {
    CAPTURE_SINK.with(|sink| {
        let mut guard = sink.borrow_mut();
        if let Some(buf) = guard.as_mut() {
            *buf = json.to_string();
            true
        } else {
            println!("{json}");
            false
        }
    })
}

#[derive(Debug)]
struct CompetitionJsonPayload {
    status: &'static str,
    reason: Option<serde_json::Value>,
    counterexample: Option<(Vec<f32>, Vec<f32>)>,
    property_file: Option<String>,
    epsilon: Option<f32>,
    threshold: f32,
    domains_explored: usize,
    domains_verified: usize,
    cuts_generated: usize,
    max_depth_reached: usize,
    time_elapsed_s: f64,
    output_bound_width: Option<f32>,
    method: Option<String>,
}

fn json_f32_array(values: &[f32]) -> Vec<serde_json::Value> {
    values.iter().map(|&value| json_f32(value)).collect()
}

/// Build a VNN-COMP standard SMT-LIB counterexample witness string.
///
/// The official VNN-COMP counterexample checker expects SMT-LIB-style variable
/// assignments rather than a raw JSON array, e.g.:
/// ```text
/// ((X_0 0.123456)
/// (X_1 -0.654321)
/// (Y_0 1.200000)
/// (Y_1 -0.300000))
/// ```
/// where `X_i` are the flattened network INPUT values (input-tensor flatten
/// order) and `Y_j` the corresponding network OUTPUT values.
///
/// Full f32 precision is preserved (no truncation) so the checker can re-run the
/// assignment through the ONNX model and reproduce the reported output.
fn counterexample_vnnlib(input: &[f32], output: &[f32]) -> String {
    let mut lines = Vec::with_capacity(input.len() + output.len());
    for (i, &value) in input.iter().enumerate() {
        lines.push(format!("(X_{i} {value})"));
    }
    for (j, &value) in output.iter().enumerate() {
        lines.push(format!("(Y_{j} {value})"));
    }
    format!("({})", lines.join("\n"))
}

fn bounded_tensor_width(result: &BetaCrownResult) -> Option<f32> {
    result.output_bounds.as_ref().and_then(|bounds| {
        let width = bounds.width().iter().copied().fold(0.0f32, f32::max);
        width.is_finite().then_some(width)
    })
}

#[cfg(feature = "mip")]
fn bounds_width(bounds: &[Bound]) -> Option<f32> {
    let width = bounds
        .iter()
        .map(|bound| bound.upper() - bound.lower())
        .fold(0.0f32, f32::max);
    width.is_finite().then_some(width)
}

fn format_competition_json(payload: CompetitionJsonPayload) -> Result<String> {
    use serde_json::json;

    let counterexample_vnnlib = payload
        .counterexample
        .as_ref()
        .map(|(input, output)| counterexample_vnnlib(input, output));

    let mut json_output = json!({
        "status": payload.status,
        "reason": payload.reason,
        "counterexample": payload.counterexample.as_ref().map(|(input, output)| {
            json!({
                "input": json_f32_array(input),
                "output": json_f32_array(output),
            })
        }),
        // VNN-COMP standard SMT-LIB witness (single string, newline-separated
        // X_i/Y_j assignments). Consumed by run_instance.sh for the witness file.
        "counterexample_vnnlib": counterexample_vnnlib,
        "property_file": payload.property_file,
        "epsilon": payload.epsilon,
        "threshold": payload.threshold,
        "domains_explored": payload.domains_explored,
        "domains_verified": payload.domains_verified,
        "cuts_generated": payload.cuts_generated,
        "max_depth_reached": payload.max_depth_reached,
        "time_elapsed_s": payload.time_elapsed_s,
        "output_bound_width": payload.output_bound_width,
    });

    if let Some(method) = payload.method {
        json_output["method"] = json!(method);
    }

    Ok(serde_json::to_string_pretty(&json_output)?)
}

/// Map a peeled-network logit vector back through the peeled terminal Sigmoid
/// (#cgan-sigmoid-peel): the witness's declared Y must match the ORIGINAL
/// graph, and the peel is exactly invertible (y = sigmoid(z), computed in f64).
fn map_output_through_sigmoid(output: &[f32]) -> Vec<f32> {
    output
        .iter()
        .map(|&z| (1.0 / (1.0 + (-f64::from(z)).exp())) as f32)
        .collect()
}

fn beta_crown_json_payload(
    result: &BetaCrownResult,
    property: Option<&Path>,
    epsilon: f32,
    effective_threshold: f32,
    sigmoid_peeled: bool,
) -> CompetitionJsonPayload {
    let (status, reason, counterexample) = match &result.result {
        BabVerificationStatus::Verified => ("verified", None, None),
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => (
            "violated",
            None,
            Some((
                counterexample.clone(),
                if sigmoid_peeled {
                    map_output_through_sigmoid(output)
                } else {
                    output.clone()
                },
            )),
        ),
        BabVerificationStatus::PotentialViolation => ("potential_violation", None, None),
        BabVerificationStatus::Unknown { reason } => {
            ("unknown", Some(serde_json::json!(reason)), None)
        }
        BabVerificationStatus::Timeout => ("timeout", None, None),
    };

    CompetitionJsonPayload {
        status,
        reason,
        counterexample,
        property_file: property.map(|path| path.display().to_string()),
        epsilon: if property.is_none() {
            Some(epsilon)
        } else {
            None
        },
        threshold: effective_threshold,
        domains_explored: result.domains_explored,
        domains_verified: result.domains_verified,
        cuts_generated: result.cuts_generated,
        max_depth_reached: result.max_depth_reached,
        time_elapsed_s: result.time_elapsed.as_secs_f64(),
        output_bound_width: bounded_tensor_width(result),
        method: None,
    }
}

#[cfg(feature = "mip")]
pub(super) fn format_verification_result_json(
    result: &VerificationResult,
    property: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: Duration,
    method: &str,
) -> Result<String> {
    let (status, reason, counterexample, output_bound_width) = match result {
        VerificationResult::Verified { output_bounds, .. } => {
            ("verified", None, None, bounds_width(output_bounds))
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => (
            "violated",
            None,
            Some((counterexample.clone(), output.clone())),
            None,
        ),
        VerificationResult::Unknown { reason, bounds, .. } => (
            "unknown",
            Some(serde_json::to_value(reason)?),
            None,
            bounds_width(bounds),
        ),
        VerificationResult::Timeout { partial_bounds, .. } => (
            "timeout",
            None,
            None,
            partial_bounds.as_deref().and_then(bounds_width),
        ),
    };

    format_competition_json(CompetitionJsonPayload {
        status,
        reason,
        counterexample,
        property_file: property.map(|path| path.display().to_string()),
        epsilon: if property.is_none() {
            Some(epsilon)
        } else {
            None
        },
        threshold,
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed_s: elapsed.as_secs_f64(),
        output_bound_width,
        method: Some(method.to_string()),
    })
}

#[cfg(feature = "mip")]
pub(super) fn verification_result_exit_code(result: &VerificationResult) -> i32 {
    match result {
        VerificationResult::Verified { .. } => exit_codes::VERIFIED,
        VerificationResult::Violated { .. } => exit_codes::VIOLATED,
        VerificationResult::Unknown { .. } => exit_codes::UNKNOWN,
        VerificationResult::Timeout { .. } => exit_codes::TIMEOUT,
    }
}

fn beta_crown_exit_code(status: &BabVerificationStatus) -> i32 {
    match status {
        BabVerificationStatus::Verified => exit_codes::VERIFIED,
        BabVerificationStatus::Violated { .. } => exit_codes::VIOLATED,
        // #3678 rewrites property-backed PotentialViolation to Violated/Unknown before
        // this renderer. The remaining surface is the propertyless threshold path, so
        // shell-level status should stay conservative rather than imply a confirmed SAT.
        BabVerificationStatus::PotentialViolation | BabVerificationStatus::Unknown { .. } => {
            exit_codes::UNKNOWN
        }
        BabVerificationStatus::Timeout => exit_codes::TIMEOUT,
    }
}

/// Output verification result.
///
/// `verify_upper` selects the direction words in the human-readable report:
/// upper-bound specs (`Y_i >= c` unsafe) prove `output < c`, lower-bound specs
/// prove `output > c`. The JSON payload carries only the numeric threshold, so
/// it does not depend on the flag.
pub(super) fn output_result(
    result: &BetaCrownResult,
    property: &Option<PathBuf>,
    epsilon: f32,
    effective_threshold: f32,
    verify_upper: bool,
    json: bool,
    sigmoid_peeled: bool,
) -> Result<()> {
    if json {
        let payload = beta_crown_json_payload(
            result,
            property.as_deref(),
            epsilon,
            effective_threshold,
            sigmoid_peeled,
        );
        let rendered = format_competition_json(payload)?;
        if emit_competition_json(&rendered) {
            // Verdict captured in-process (vnncomp path): do NOT print or exit;
            // the caller reads it via `take_captured_json` and translates it.
            return Ok(());
        }
    } else {
        // Mirrors the "Threshold: ... (verifying ...)" line printed at dispatch
        // time: a proven upper bound means every output is BELOW the threshold
        // and a counterexample is one at or above it.
        let (proven, violated, potential) = if verify_upper {
            ("<", ">=", ">=")
        } else {
            (">", "<=", "<")
        };
        println!("\n--- Result ---");
        match &result.result {
            BabVerificationStatus::Verified => {
                println!("Status: VERIFIED");
                println!(
                    "All inputs produce output {} {}",
                    proven, effective_threshold
                );
            }
            BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                println!("Status: VIOLATED");
                println!(
                    "Found counterexample where output {} {}",
                    violated, effective_threshold
                );
                println!("Counterexample input: {:?}", counterexample);
                println!("Counterexample output: {:?}", output);
            }
            BabVerificationStatus::PotentialViolation => {
                println!("Status: POTENTIAL VIOLATION");
                println!(
                    "Found region where output may be {} {}",
                    potential, effective_threshold
                );
            }
            BabVerificationStatus::Unknown { reason } => {
                println!("Status: UNKNOWN");
                println!("Reason: {}", reason);
            }
            BabVerificationStatus::Timeout => {
                println!("Status: TIMEOUT");
                println!("Verification timed out before completion");
            }
        }
        println!("Domains explored: {}", result.domains_explored);
        println!("Domains verified: {}", result.domains_verified);
        if result.cuts_generated > 0 {
            println!("Cuts generated: {}", result.cuts_generated);
        }
        println!("Max depth reached: {}", result.max_depth_reached);
        println!("Time elapsed: {:.2}s", result.time_elapsed.as_secs_f64());

        if let Some(bounds) = &result.output_bounds {
            let width = bounds.width();
            let max_width = width.iter().cloned().fold(0.0f32, f32::max);
            println!("Output bound width: {:.6e}", max_width);
        }
    }

    // Apply exit codes matching the verify command contract.
    // Per designs/2026-02-03-smt-result-semantics.md.
    let exit_code = beta_crown_exit_code(&result.result);
    if exit_code != exit_codes::VERIFIED {
        #[cfg(test)]
        return Ok(());
        #[cfg(not(test))]
        std::process::exit(exit_code);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_format_beta_crown_json_nests_counterexample_3708() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: vec![0.5],
                    output: vec![-1.0],
                },
                domains_explored: 3,
                time_elapsed: Duration::from_secs(2),
                max_depth_reached: 1,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            },
            None,
            0.02,
            0.0,
            false,
        );
        let json =
            format_competition_json(payload).expect("beta-crown JSON payload should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("beta-crown JSON must parse");

        assert_eq!(parsed["status"], "violated");
        assert_eq!(parsed["counterexample"]["input"][0], 0.5);
        assert_eq!(parsed["counterexample"]["output"][0], -1.0);
        assert!(parsed.get("output").is_none(), "output must stay nested");
    }

    #[test]
    fn test_counterexample_vnnlib_smtlib_format() {
        // Synthetic 2-input / 2-output counterexample.
        let smt = counterexample_vnnlib(&[0.123_456_f32, -0.654_321_f32], &[1.2_f32, -0.3_f32]);

        // Outer parentheses wrap the whole assignment list.
        assert!(smt.starts_with('('), "must start with '(': {smt}");
        assert!(smt.ends_with(')'), "must end with ')': {smt}");

        // One assignment per line; inputs are X_i, outputs are Y_j, in order.
        let lines: Vec<&str> = smt.lines().collect();
        assert_eq!(lines.len(), 4, "one assignment per line: {smt}");
        assert!(lines[0].starts_with("((X_0 "), "first input X_0: {smt}");
        assert!(lines[1].starts_with("(X_1 "), "second input X_1: {smt}");
        assert!(lines[2].starts_with("(Y_0 "), "first output Y_0: {smt}");
        assert!(lines[3].starts_with("(Y_1 "), "second output Y_1: {smt}");

        // Each line is a balanced parenthesised assignment.
        for line in &lines {
            assert!(line.contains('('), "assignment has '(': {line}");
            assert!(line.ends_with(')'), "assignment ends ')': {line}");
        }

        // Full precision retained (not truncated to 6 decimals).
        assert!(smt.contains("0.123456"), "input precision preserved: {smt}");
        assert!(
            smt.contains("-0.654321"),
            "negative input precision preserved: {smt}"
        );
    }

    #[test]
    fn test_format_beta_crown_json_includes_vnnlib_witness() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: vec![0.5, -0.25],
                    output: vec![-1.0, 2.0],
                },
                domains_explored: 1,
                time_elapsed: Duration::from_secs(1),
                max_depth_reached: 1,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 0,
            },
            None,
            0.02,
            0.0,
            false,
        );
        let json =
            format_competition_json(payload).expect("beta-crown JSON payload should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("beta-crown JSON must parse");

        let witness = parsed["counterexample_vnnlib"]
            .as_str()
            .expect("counterexample_vnnlib must be a string");
        assert!(witness.contains("(X_0 "), "witness has X_0: {witness}");
        assert!(witness.contains("(X_1 "), "witness has X_1: {witness}");
        assert!(witness.contains("(Y_0 "), "witness has Y_0: {witness}");
        assert!(witness.contains("(Y_1 "), "witness has Y_1: {witness}");
    }

    #[test]
    fn test_format_beta_crown_json_omits_vnnlib_when_no_counterexample() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: Duration::from_secs(1),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            },
            None,
            0.02,
            0.0,
            false,
        );
        let json = format_competition_json(payload).expect("verified JSON should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("verified JSON must parse");
        assert!(
            parsed["counterexample_vnnlib"].is_null(),
            "no witness when verified: {json}"
        );
    }

    #[test]
    fn test_potential_violation_exit_code_is_unknown_3708() {
        assert_eq!(
            beta_crown_exit_code(&BabVerificationStatus::PotentialViolation),
            exit_codes::UNKNOWN
        );
    }

    // Tests for the SMT/MIP-specific format_verification_result_json are gated
    // behind the smt/mip features since the function itself is cfg-gated.
    #[cfg(feature = "mip")]
    mod verification_result_json_tests {
        use super::super::*;
        use ny_core::{Bound, SoundnessProvenance, UnknownReason, VerificationResult};
        use std::time::Duration;

        fn violated_verification_result() -> VerificationResult {
            VerificationResult::Violated {
                provenance: SoundnessProvenance::sound(),
                counterexample: vec![0.25, 0.75],
                output: vec![1.0, -2.0],
                details: None,
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }

        #[test]
        fn test_format_verification_result_json_nests_counterexample_3708() {
            let json = format_verification_result_json(
                &violated_verification_result(),
                None,
                0.01,
                0.0,
                Duration::from_millis(1500),
                "mip-highs",
            )
            .expect("verification result JSON should serialize");

            assert!(
                json.contains("\"status\": \"violated\""),
                "status field should match run_instance grep contract: {json}"
            );
            assert!(
                !json.contains("\\\"status\\\""),
                "status field must not be backslash-escaped: {json}"
            );

            let parsed: serde_json::Value =
                serde_json::from_str(&json).expect("verification result JSON must parse");
            assert_eq!(parsed["status"], "violated");
            assert_eq!(parsed["method"], "mip-highs");
            assert_eq!(parsed["domains_explored"], 0);
            assert!(parsed.get("output").is_none(), "output must stay nested");
            assert_eq!(parsed["counterexample"]["input"][0], 0.25);
            assert_eq!(parsed["counterexample"]["output"][1], -2.0);
        }

        #[test]
        fn test_format_verification_result_json_preserves_structured_reason_3708() {
            let json = format_verification_result_json(
                &VerificationResult::Unknown {
                    provenance: SoundnessProvenance::sound(),
                    bounds: vec![Bound::new(0.0, 1.0)],
                    reason: UnknownReason::SmtUnknown {
                        solver_reason: Some("ay returned unknown".to_string()),
                    },
                    actual_method: Some(ny_core::MethodUsed::Mip),
                },
                None,
                0.01,
                0.0,
                Duration::from_secs(1),
                "mip",
            )
            .expect("unknown JSON should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&json).expect("unknown JSON must parse");

            assert_eq!(parsed["status"], "unknown");
            assert_eq!(parsed["reason"]["type"], "smt_unknown");
            assert_eq!(parsed["reason"]["solver_reason"], "ay returned unknown");
        }
    }
}
