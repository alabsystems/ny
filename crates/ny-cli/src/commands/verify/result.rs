// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification result rendering (JSON/text) and exit-code policy.

use anyhow::Result;
use ny_core::{Bound, MethodUsed, VerificationSpec};
use ny_propagate::soundness::{
    count_sqrt_negative_domain_graph, count_sqrt_negative_domain_network,
};
use ny_propagate::{PropagationMethod, Verifier};

use std::path::Path;

use super::super::JsonCliError;
use super::model_load::VerifiableNetwork;
use crate::commands::backend::ProofBackendReceipt;
use crate::commands::terminal_peel::AppliedTerminalPeel;

/// Convert an `f32` to a `serde_json::Value`, encoding non-finite values as strings.
///
/// JSON has no representation for Infinity or NaN. `serde_json::json!()` silently
/// converts these to `null`, losing the distinction between "infinite bound" and
/// "missing value". This function renders them as string values instead (#3133):
/// - `f32::INFINITY` → `"Infinity"`
/// - `f32::NEG_INFINITY` → `"-Infinity"`
/// - `f32::NAN` → `"NaN"`
/// - Finite values → `serde_json::Value::Number`
pub(crate) fn json_f32(v: f32) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else if v.is_nan() {
        serde_json::Value::String("NaN".to_string())
    } else if v.is_sign_positive() {
        serde_json::Value::String("Infinity".to_string())
    } else {
        serde_json::Value::String("-Infinity".to_string())
    }
}

/// Stable JSON evidence for the backend request, live qualification, and any
/// fallback. Keeping this as one shared projection prevents the standard,
/// diagnostic, and f64 verify renderers from disagreeing about execution.
pub(super) fn backend_receipt_json(receipt: &ProofBackendReceipt) -> serde_json::Value {
    serde_json::json!({
        "requested": receipt.requested.to_string(),
        "request_source": receipt.request_source.as_str(),
        "selection_reason": receipt.selection_reason.as_deref(),
        "effective": receipt.effective.to_string(),
        "qualification": receipt.qualification.as_str(),
        "qualification_provenance": receipt.qualification_provenance.as_deref(),
        "failed_rung": receipt.failed_rung.as_deref(),
        "fallback_reason": receipt.fallback_reason.as_deref(),
        "provenance": receipt.provenance.as_str(),
    })
}

/// Exit codes for verification results.
///
/// Per design doc: designs/2026-02-03-smt-result-semantics.md
pub(crate) mod exit_codes {
    /// Property holds - verification successful.
    pub(crate) const VERIFIED: i32 = 0;
    /// Property violated - counterexample found.
    pub(crate) const VIOLATED: i32 = 1;
    /// Inconclusive - solver couldn't decide.
    pub(crate) const UNKNOWN: i32 = 2;
    /// Time limit exceeded.
    pub(crate) const TIMEOUT: i32 = 3;
    /// Invalid invocation, configuration/load failure, unsupported request, or
    /// internal operational error. Verdict codes 0-3 are reserved exclusively
    /// for completed verification outcomes.
    pub(crate) const ERROR: i32 = 4;
}

/// Check for sqrt-negative-domain nodes when `--require-sound` is active.
pub(crate) fn check_require_sound_sqrt(
    network: &VerifiableNetwork,
    spec: &VerificationSpec,
) -> Result<()> {
    let input_bounds = Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;
    let sqrt_negative_domain_nodes = match network {
        VerifiableNetwork::Sequential(net) => {
            count_sqrt_negative_domain_network(net, &input_bounds)?
        }
        VerifiableNetwork::Graph(graph) => count_sqrt_negative_domain_graph(graph, &input_bounds)?,
    };
    if sqrt_negative_domain_nodes > 0 {
        let error_output = serde_json::json!({
            "error": "require_sound",
            "message": "Sqrt input bounds include negative values; require-sound forbids negative-domain sqrt.",
            "sqrt_negative_domain_nodes": sqrt_negative_domain_nodes,
        });
        return Err(JsonCliError::from_value(error_output).into());
    }
    Ok(())
}

/// Run standard verification and render the result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_standard_verification(
    network: &VerifiableNetwork,
    spec: &VerificationSpec,
    verifier: &Verifier,
    requested_method: PropagationMethod,
    backend_receipt: &ProofBackendReceipt,
    epsilon: f32,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
    property: Option<&Path>,
    strict: bool,
    require_sound: bool,
    allow_unknown: bool,
    json: bool,
    applied_terminal_peel: AppliedTerminalPeel,
) -> Result<()> {
    // #margin-subset-seed: tell the bound collection which OUTPUT rows the
    // verdict can actually read, for the scope of this verification.
    //
    // Without this the collector seeds a full `[output_dim x output_dim]`
    // identity at the OUTPUT node even when the property names a handful of
    // outputs. On TinyYOLO (yolo_2023) that is a 3.57 GB identity pair for 5 of
    // 21,125 outputs — an allocation the Conv2d scratch cap refuses, so the
    // node degrades to loose IBP for want of rows nothing reads.
    //
    // Held across the verify call because the publication is thread-local and
    // scoped; dropping it restores the previous (usually absent) publication.
    //
    // Sound: this only selects which rows get the tighter treatment. Unselected
    // rows keep their valid IBP/forward enclosures, so a too-small set costs
    // tightness and never validity, and an empty set is byte-identical to the
    // historical full-width path.
    let _output_seed_scope = vnnlib_spec.map(|parsed| {
        let referenced = parsed.referenced_output_indices();
        tracing::info!(
            "#margin-subset-seed: verify publishing {} of {} spec-referenced OUTPUT indices",
            referenced.len(),
            parsed.num_outputs,
        );
        ny_propagate::SpecOutputSeedScope::publish(referenced)
    });

    let result = match network {
        VerifiableNetwork::Sequential(net) => verifier.verify(net, spec)?,
        VerifiableNetwork::Graph(graph) => verifier.verify_graph(graph, spec)?,
    };

    // Check strict mode: fail if method fallback occurred
    if strict {
        if let Some(actual_tag) = result.actual_method_tag() {
            if !actual_method_matches_requested(actual_tag, requested_method) {
                anyhow::bail!(
                    "Strict mode: requested method '{:?}' but fell back to '{}'. \
                    Remove --strict to allow fallback.",
                    requested_method,
                    actual_tag.as_str()
                );
            }
        }
    }

    // Check require-sound mode: fail if heuristics were used
    if require_sound {
        let provenance = result.provenance();
        if provenance.mode() == ny_core::VerificationSoundnessMode::Heuristic {
            let status = match &result {
                ny_core::VerificationResult::Verified { .. } => "verified",
                ny_core::VerificationResult::Violated { .. } => "violated",
                ny_core::VerificationResult::Unknown { .. } => "unknown",
                ny_core::VerificationResult::Timeout { .. } => "timeout",
            };
            let soundness = serde_json::to_value(provenance)?;
            let error_output = serde_json::json!({
                "error": "require_sound",
                "message": "Verification used heuristics and is not provably sound",
                "verification_status": status,
                "soundness": soundness,
            });
            return Err(JsonCliError::from_value(error_output).into());
        }
    }

    // Check property constraints if VNN-LIB spec is provided
    let property_status = evaluate_property_status(&result, vnnlib_spec)?;

    // Render output
    let publication_refused = if json {
        render_json(
            &result,
            requested_method,
            backend_receipt,
            epsilon,
            property_status,
            property,
            applied_terminal_peel,
        )?
    } else {
        render_text(&result, property_status, applied_terminal_peel)?
    };

    // Determine and apply exit code
    let exit_code =
        compute_publication_exit_code(&result, property_status, allow_unknown, publication_refused);
    if exit_code != exit_codes::VERIFIED {
        std::process::exit(exit_code);
    }

    Ok(())
}

/// Evaluate VNN-LIB property status against output bounds.
///
/// Returns `Ok(None)` when no VNN-LIB spec is provided or no constraints exist.
/// Returns `Err` when a constraint references an output index beyond the bounds
/// vector length (malformed VNN-LIB property file).
pub(super) fn evaluate_property_status(
    result: &ny_core::VerificationResult,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
) -> Result<Option<&'static str>> {
    let vnnlib = match vnnlib_spec {
        Some(v) => v,
        None => return Ok(None),
    };
    use ny_onnx::vnnlib::OutputConstraint;

    fn get_valid_bound(bounds: &[Bound], idx: usize) -> Result<Option<&Bound>> {
        let bound = bounds.get(idx).ok_or_else(|| {
            anyhow::anyhow!(
                "VNN-LIB constraint references output index {} but model has {} outputs",
                idx,
                bounds.len()
            )
        })?;
        Ok((bound.lower().is_finite()
            && bound.upper().is_finite()
            && bound.lower() <= bound.upper())
        .then_some(bound))
    }

    let check_constraint_satisfiable =
        |bounds: &[Bound], constraint: &OutputConstraint| -> Result<bool> {
            Ok(match constraint {
                OutputConstraint::LessEq(i, j) => {
                    match (get_valid_bound(bounds, *i)?, get_valid_bound(bounds, *j)?) {
                        (Some(left), Some(right)) => left.lower() <= right.upper(),
                        _ => true,
                    }
                }
                OutputConstraint::GreaterEq(i, j) => {
                    match (get_valid_bound(bounds, *i)?, get_valid_bound(bounds, *j)?) {
                        (Some(left), Some(right)) => left.upper() >= right.lower(),
                        _ => true,
                    }
                }
                OutputConstraint::LessThan(i, j) => {
                    match (get_valid_bound(bounds, *i)?, get_valid_bound(bounds, *j)?) {
                        (Some(left), Some(right)) => left.lower() < right.upper(),
                        _ => true,
                    }
                }
                OutputConstraint::GreaterThan(i, j) => {
                    match (get_valid_bound(bounds, *i)?, get_valid_bound(bounds, *j)?) {
                        (Some(left), Some(right)) => left.upper() > right.lower(),
                        _ => true,
                    }
                }
                OutputConstraint::LessEqConst(i, c) => {
                    get_valid_bound(bounds, *i)?.is_none_or(|bound| f64::from(bound.lower()) <= *c)
                }
                OutputConstraint::GreaterEqConst(i, c) => {
                    get_valid_bound(bounds, *i)?.is_none_or(|bound| f64::from(bound.upper()) >= *c)
                }
                OutputConstraint::LessThanConst(i, c) => {
                    get_valid_bound(bounds, *i)?.is_none_or(|bound| f64::from(bound.lower()) < *c)
                }
                OutputConstraint::GreaterThanConst(i, c) => {
                    get_valid_bound(bounds, *i)?.is_none_or(|bound| f64::from(bound.upper()) > *c)
                }
                _ => true, // conservatively assume unknown variants are satisfiable
            })
        };

    match result {
        ny_core::VerificationResult::Verified { output_bounds, .. }
        | ny_core::VerificationResult::Unknown {
            bounds: output_bounds,
            ..
        } => {
            let clauses: Vec<&[OutputConstraint]> = if vnnlib.output_constraint_clauses.is_empty() {
                if vnnlib.output_constraints.is_empty() {
                    Vec::new()
                } else {
                    vec![vnnlib.output_constraints.as_slice()]
                }
            } else {
                vnnlib
                    .output_constraint_clauses
                    .iter()
                    .map(|c| c.as_slice())
                    .collect()
            };

            if clauses.is_empty() {
                Ok(None)
            } else {
                // Validate exactly the output rows that can affect the
                // property. A malformed referenced interval makes the whole
                // property unknown; an unused conservative row is irrelevant.
                for constraint in clauses.iter().flat_map(|clause| clause.iter()) {
                    let indices = match constraint {
                        OutputConstraint::LessEq(i, j)
                        | OutputConstraint::LessThan(i, j)
                        | OutputConstraint::GreaterEq(i, j)
                        | OutputConstraint::GreaterThan(i, j) => [Some(*i), Some(*j)],
                        OutputConstraint::LessEqConst(i, _)
                        | OutputConstraint::LessThanConst(i, _)
                        | OutputConstraint::GreaterEqConst(i, _)
                        | OutputConstraint::GreaterThanConst(i, _) => [Some(*i), None],
                        _ => [None, None],
                    };
                    for index in indices.into_iter().flatten() {
                        if get_valid_bound(output_bounds, index)?.is_none() {
                            return Ok(Some("unknown"));
                        }
                    }
                }
                if clauses.iter().flat_map(|clause| clause.iter()).any(
                    |constraint| match constraint {
                        OutputConstraint::LessEqConst(_, c)
                        | OutputConstraint::LessThanConst(_, c)
                        | OutputConstraint::GreaterEqConst(_, c)
                        | OutputConstraint::GreaterThanConst(_, c) => !c.is_finite(),
                        _ => false,
                    },
                ) {
                    return Ok(Some("unknown"));
                }

                // Check each clause; propagate index-out-of-bounds errors.
                let mut clause_satisfiable = Vec::with_capacity(clauses.len());
                for clause in &clauses {
                    let mut all_satisfied = true;
                    for c in *clause {
                        if !check_constraint_satisfiable(output_bounds, c)? {
                            all_satisfied = false;
                            break;
                        }
                    }
                    clause_satisfiable.push(all_satisfied);
                }

                // Each clause is a conjunction. A disjunctive unsafe region can
                // still hold when ANY clause can hold; a conjunctive unsafe
                // region can still hold only when EVERY clause can hold.
                let unsafe_region_may_hold = if vnnlib.is_disjunction {
                    clause_satisfiable.iter().any(|&satisfiable| satisfiable)
                } else {
                    clause_satisfiable.iter().all(|&satisfiable| satisfiable)
                };
                if unsafe_region_may_hold {
                    Ok(Some("unknown"))
                } else {
                    Ok(Some("safe"))
                }
            }
        }
        _ => Ok(None),
    }
}

/// Build the JSON result and report whether a peeled witness had to be refused.
fn verification_json_value(
    result: &ny_core::VerificationResult,
    requested_method: PropagationMethod,
    backend_receipt: &ProofBackendReceipt,
    epsilon: f32,
    property_status: Option<&str>,
    property: Option<&Path>,
    applied_terminal_peel: AppliedTerminalPeel,
) -> Result<(serde_json::Value, bool)> {
    use serde_json::json;
    let publication_refused = applied_terminal_peel.needs_original_output_rehydration()
        && matches!(result, ny_core::VerificationResult::Violated { .. });
    let method_str = match requested_method {
        PropagationMethod::Ibp => "ibp",
        PropagationMethod::Crown => "crown",
        PropagationMethod::AlphaCrown => "alpha",
        PropagationMethod::SdpCrown => "sdp-crown",
        PropagationMethod::BetaCrown => "beta",
    };
    let mut output = match result {
        ny_core::VerificationResult::Verified {
            output_bounds,
            proof,
            actual_method,
            ..
        } => {
            let mut r = json!({
                "status": "verified",
                "output_bounds": output_bounds.iter().map(|b| {
                    json!({"lower": json_f32(b.lower()), "upper": json_f32(b.upper())})
                }).collect::<Vec<_>>(),
                "epsilon": epsilon,
                "method": method_str,
                "backend": backend_receipt.effective.to_string()
            });
            if let Some(am) = actual_method {
                if let Some(obj) = r.as_object_mut() {
                    obj.insert("actual_method".to_string(), json!(am));
                }
            }
            if let Some(proof) = proof {
                let proof: &ny_core::VerificationProof = proof;
                if let Some(obj) = r.as_object_mut() {
                    obj.insert(
                        "proof".to_string(),
                        json!({
                            "format": format!("{:?}", proof.format()),
                            "num_steps": proof.num_steps(),
                            "size_bytes": proof.as_bytes().len()
                        }),
                    );
                }
            }
            r
        }
        ny_core::VerificationResult::Violated {
            counterexample,
            output,
            details,
            actual_method,
            ..
        } => {
            let mut json_val = if publication_refused {
                json!({
                    "status": "unknown",
                    "reason": format!(
                        "peeled {} counterexample requires original-model output rehydration",
                        applied_terminal_peel.activation_name()
                    ),
                    "epsilon": epsilon,
                    "method": method_str,
                    "backend": backend_receipt.effective.to_string()
                })
            } else {
                let published_output = applied_terminal_peel
                    .output_in_original_coordinates(output)
                    .expect("non-Softmax-family output mapping must be defined");
                json!({
                    "status": "violated",
                    "counterexample": counterexample,
                    "output": &*published_output,
                    "output_coordinates": "original_model",
                    "epsilon": epsilon,
                    "method": method_str,
                    "backend": backend_receipt.effective.to_string()
                })
            };
            if let Some(am) = actual_method {
                if let Some(obj) = json_val.as_object_mut() {
                    obj.insert("actual_method".to_string(), json!(am));
                }
            }
            // InformativeCounterexample values and rewritten thresholds are in
            // peeled coordinates. Do not mix those with a Sigmoid-rehydrated
            // output vector, and never publish them for a refused Softmax seed.
            if !applied_terminal_peel.applied() {
                if let Some(ref details) = details {
                    if let Some(vc) = details.violated_constraint() {
                        json_val["violation"] = json!({
                            "output_idx": vc.output_idx(),
                            "actual_value": json_f32(vc.actual_value()),
                            "required_lower": json_f32(vc.required_bound().lower()),
                            "required_upper": json_f32(vc.required_bound().upper()),
                            "violation_amount": json_f32(vc.violation_amount()),
                            "explanation": vc.explain()
                        });
                    }
                    json_val["explanation"] = json!(details.explanation());
                }
            } else if !publication_refused {
                json_val["constraint_coordinates"] = json!("peeled_preactivation");
            }
            json_val
        }
        ny_core::VerificationResult::Unknown {
            reason,
            bounds,
            actual_method,
            ..
        } => {
            let mut json_val = json!({
                "status": "unknown",
                "reason": reason,
                "output_bounds": bounds.iter().map(|b| {
                    json!({"lower": json_f32(b.lower()), "upper": json_f32(b.upper())})
                }).collect::<Vec<_>>(),
                "epsilon": epsilon,
                "method": method_str,
                "backend": backend_receipt.effective.to_string()
            });
            if let Some(am) = actual_method {
                if let Some(obj) = json_val.as_object_mut() {
                    obj.insert("actual_method".to_string(), json!(am));
                }
            }
            json_val
        }
        ny_core::VerificationResult::Timeout {
            partial_bounds,
            actual_method,
            ..
        } => {
            let mut json_val = json!({
                "status": "timeout",
                "partial_bounds": partial_bounds.as_ref().map(|bs| {
                    bs.iter().map(|b| {
                        json!({"lower": json_f32(b.lower()), "upper": json_f32(b.upper())})
                    }).collect::<Vec<_>>()
                }),
                "epsilon": epsilon,
                "method": method_str,
                "backend": backend_receipt.effective.to_string()
            });
            if let Some(am) = actual_method {
                if let Some(obj) = json_val.as_object_mut() {
                    obj.insert("actual_method".to_string(), json!(am));
                }
            }
            json_val
        }
    };

    if let Some(status) = property_status {
        output["property_status"] = json!(status);
        // Fix: property_status overrides top-level "status" for Verified results.
        // VerificationResult::Verified means "bound propagation completed" — it does NOT
        // mean the property was proven safe. When property_status is "unknown", the bounds
        // are too loose to prove the property. Only "safe" means genuinely verified.
        // This mirrors compute_exit_code() which already handles this correctly (#1456).
        if matches!(result, ny_core::VerificationResult::Verified { .. }) {
            match status {
                "safe" => { /* status already "verified" — correct */ }
                "violated" => {
                    output["status"] = json!("violated");
                }
                _ => {
                    // "unknown" or any other non-safe status: bounds don't prove property
                    output["status"] = json!("unknown");
                }
            }
        }
    }
    if let Some(p) = property {
        output["property_file"] = json!(p.display().to_string());
    }

    output["soundness"] = serde_json::to_value(result.provenance())?;
    output["backend_receipt"] = backend_receipt_json(backend_receipt);
    output["terminal_peel"] = json!({
        "applied": applied_terminal_peel.applied(),
        "activation": applied_terminal_peel,
    });
    output["output_coordinates"] = json!(if matches!(
        result,
        ny_core::VerificationResult::Violated { .. }
    ) && !publication_refused
    {
        "original_model"
    } else if applied_terminal_peel.applied() {
        "peeled_preactivation"
    } else {
        "original_model"
    });

    Ok((output, publication_refused))
}

/// Render verification result as JSON.
fn render_json(
    result: &ny_core::VerificationResult,
    requested_method: PropagationMethod,
    backend_receipt: &ProofBackendReceipt,
    epsilon: f32,
    property_status: Option<&str>,
    property: Option<&Path>,
    applied_terminal_peel: AppliedTerminalPeel,
) -> Result<bool> {
    let (output, publication_refused) = verification_json_value(
        result,
        requested_method,
        backend_receipt,
        epsilon,
        property_status,
        property,
        applied_terminal_peel,
    )?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(publication_refused)
}

/// Render verification result as human-readable text.
fn render_text(
    result: &ny_core::VerificationResult,
    property_status: Option<&str>,
    applied_terminal_peel: AppliedTerminalPeel,
) -> Result<bool> {
    println!("\nVerification Result:");
    let (summary, publication_refused) = verification_result_text(result, applied_terminal_peel);
    println!("{summary}");

    let mode_str = match result.provenance().mode() {
        ny_core::VerificationSoundnessMode::Sound => "sound",
        ny_core::VerificationSoundnessMode::Heuristic => "heuristic",
    };
    if !result.provenance().heuristics_used().is_empty() {
        println!(
            "\nSoundness: {} (heuristics used: {})",
            mode_str,
            result.provenance().heuristics_used().len()
        );
        for h in result.provenance().heuristics_used() {
            println!("  - {:?}", h);
        }
    } else {
        println!("\nSoundness: {}", mode_str);
    }

    if let Some(status) = property_status {
        println!("\nProperty Status: {}", status.to_uppercase());
        if status == "safe" {
            println!("  The output bounds prove the property CANNOT be violated.");
        } else {
            println!("  The output bounds do not prove safety. Property may be violated.");
        }
    }
    Ok(publication_refused)
}

fn verification_result_text(
    result: &ny_core::VerificationResult,
    applied_terminal_peel: AppliedTerminalPeel,
) -> (String, bool) {
    match result {
        ny_core::VerificationResult::Violated { .. }
            if applied_terminal_peel.needs_original_output_rehydration() =>
        {
            (
                format!(
                    "Status: UNKNOWN\nReason: peeled {} counterexample output is a preactivation, \
                 not original-model Y; refusing witness publication",
                    applied_terminal_peel.activation_name()
                ),
                true,
            )
        }
        ny_core::VerificationResult::Violated {
            counterexample,
            output,
            ..
        } if applied_terminal_peel.is_sigmoid() => {
            let (label, published_output) = applied_terminal_peel.human_witness_output(output);
            (
                format!(
                    "Status: VIOLATED\nCounterexample input: {counterexample:?}\n\
                     {label}: {published_output:?}"
                ),
                false,
            )
        }
        _ if applied_terminal_peel.applied() => (
            format!(
                "Output-coordinate note: result bounds are preactivations before the peeled {}; \
                 they are not original-model Y bounds.\n{result:?}",
                applied_terminal_peel.activation_name()
            ),
            false,
        ),
        // `VerificationResult::Verified` means BOUND COMPUTATION SUCCEEDED, not
        // "the property holds" -- the verdict is the separately-computed
        // `property_status`, printed below as "Property Status". Debug-printing the
        // bare variant put the word "Verified" at the top of the output while the
        // real verdict two lines down said UNKNOWN. That misreading has already
        // caused a soundness bug to be reported against a run that was correct, so
        // the variant is never allowed to speak for itself here.
        ny_core::VerificationResult::Verified { output_bounds, .. } => (
            format!(
                "Bound computation: COMPLETE ({} sound output bounds).\n\
                 This is NOT the property verdict -- see \"Property Status\" below.\n\
                 {result:?}",
                output_bounds.len()
            ),
            false,
        ),
        _ => (format!("{result:?}"), false),
    }
}

/// Compute the exit code based on verification result and property status.
///
/// Per design doc: designs/2026-02-03-smt-result-semantics.md
/// Fix for #1456: property_status overrides VerificationResult for exit code
pub(super) fn compute_exit_code(
    result: &ny_core::VerificationResult,
    property_status: Option<&str>,
    allow_unknown: bool,
) -> i32 {
    match result {
        ny_core::VerificationResult::Verified { .. } => match property_status {
            Some("safe") => exit_codes::VERIFIED,
            Some("unknown") => {
                if allow_unknown {
                    exit_codes::VERIFIED
                } else {
                    exit_codes::UNKNOWN
                }
            }
            None => exit_codes::VERIFIED,
            Some(_) => exit_codes::UNKNOWN,
        },
        ny_core::VerificationResult::Violated { .. } => exit_codes::VIOLATED,
        ny_core::VerificationResult::Unknown { .. } => {
            if allow_unknown {
                exit_codes::VERIFIED
            } else {
                exit_codes::UNKNOWN
            }
        }
        // `--allow-unknown` is deliberately narrow: it lets CI accept a
        // completed Unknown result, but it must not hide an exhausted
        // wall-clock budget as success.
        ny_core::VerificationResult::Timeout { .. } => exit_codes::TIMEOUT,
    }
}

fn compute_publication_exit_code(
    result: &ny_core::VerificationResult,
    property_status: Option<&str>,
    allow_unknown: bool,
    publication_refused: bool,
) -> i32 {
    if publication_refused {
        // A refused witness is never SAT, and `--allow-unknown` must not turn
        // that publication failure into process success.
        exit_codes::UNKNOWN
    } else {
        compute_exit_code(result, property_status, allow_unknown)
    }
}

/// Check whether the actual verification method matches the requested propagation
/// method. Used by strict mode to reject method fallbacks without string comparison.
///
/// The f64 engine tags its runs `IbpF64`/`CrownF64`; those are the requested method
/// carried out in double precision, not a fallback. `CrownF64` is fixed-slope, so it
/// deliberately does not match a request for AlphaCrown/SdpCrown/BetaCrown.
pub(super) fn actual_method_matches_requested(
    actual: &MethodUsed,
    requested: PropagationMethod,
) -> bool {
    matches!(
        (actual, requested),
        (MethodUsed::Ibp, PropagationMethod::Ibp)
            | (MethodUsed::IbpF64, PropagationMethod::Ibp)
            | (MethodUsed::Crown, PropagationMethod::Crown)
            | (MethodUsed::CrownF64, PropagationMethod::Crown)
            | (MethodUsed::AlphaCrown, PropagationMethod::AlphaCrown)
            | (MethodUsed::SdpCrown, PropagationMethod::SdpCrown)
            | (MethodUsed::BetaCrown, PropagationMethod::BetaCrown)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::backend::{BackendRequest, BackendRequestSource};
    use crate::BackendArg;
    use ny_core::{Bound, SoundnessProvenance, VerificationResult};
    use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

    /// Helper: build a Verified result with the given output bounds.
    fn verified_with_bounds(bounds: Vec<Bound>) -> VerificationResult {
        VerificationResult::Verified {
            provenance: SoundnessProvenance::sound(),
            output_bounds: bounds,
            proof: None,
            actual_method: None,
        }
    }

    fn violated_with_output(output: Vec<f32>) -> VerificationResult {
        let counterexample = vec![0.125];
        let required = vec![Bound::new(1.0, 2.0); output.len()];
        let details = ny_core::InformativeCounterexample::new(
            counterexample.clone(),
            output.clone(),
            Some(&required),
        );
        VerificationResult::Violated {
            provenance: SoundnessProvenance::sound(),
            counterexample,
            output,
            details: Some(Box::new(details)),
            actual_method: Some(MethodUsed::Crown),
        }
    }

    fn cpu_backend_receipt() -> ProofBackendReceipt {
        ProofBackendReceipt::cpu(
            BackendRequest {
                backend: BackendArg::Cpu,
                source: BackendRequestSource::DefaultedCliValue,
                selection_reason: None,
            },
            "cpu",
        )
    }

    #[test]
    fn verification_json_preserves_backend_request_and_fallback_receipt() {
        let receipt = ProofBackendReceipt::refused_wgpu(
            BackendRequest {
                backend: BackendArg::Wgpu,
                source: BackendRequestSource::LegacyGpuFlag,
                selection_reason: None,
            },
            "cpu",
            Some("adapter-17".to_string()),
            Some("sentinel_taint".to_string()),
            "sentinel self-check refused",
        );
        let (json, refused) = verification_json_value(
            &verified_with_bounds(vec![Bound::new(0.0, 1.0)]),
            PropagationMethod::Crown,
            &receipt,
            0.1,
            None,
            None,
            AppliedTerminalPeel::None,
        )
        .expect("verification JSON");
        assert!(!refused);
        assert_eq!(json["backend"], "cpu");
        assert_eq!(json["backend_receipt"]["requested"], "wgpu");
        assert_eq!(json["backend_receipt"]["effective"], "cpu");
        assert_eq!(json["backend_receipt"]["qualification"], "refused");
        assert_eq!(
            json["backend_receipt"]["fallback_reason"],
            "sentinel self-check refused"
        );
        assert_eq!(json["backend_receipt"]["failed_rung"], "sentinel_taint");
    }

    #[test]
    fn verify_sigmoid_peel_publishes_only_rehydrated_witness_output() {
        let result = violated_with_output(vec![0.0]);
        let (text, refused) = verification_result_text(&result, AppliedTerminalPeel::Sigmoid);
        assert!(!refused);
        assert!(text.contains("Status: VIOLATED"));
        assert!(text.contains("Counterexample output: [0.5]"));

        let (json, refused) = verification_json_value(
            &result,
            PropagationMethod::Crown,
            &cpu_backend_receipt(),
            0.0,
            None,
            None,
            AppliedTerminalPeel::Sigmoid,
        )
        .expect("Sigmoid result JSON");
        assert!(!refused);
        assert_eq!(json["status"], "violated");
        assert_eq!(json["output"], serde_json::json!([0.5]));
        assert_eq!(json["output_coordinates"], "original_model");
        assert_eq!(json["terminal_peel"]["activation"], "sigmoid");
        assert!(json.get("violation").is_none());
        assert!(json.get("explanation").is_none());
    }

    #[test]
    fn verify_softmax_family_peel_refuses_original_output_witness_publication() {
        let result = violated_with_output(vec![42.25]);
        for peel in [
            AppliedTerminalPeel::Softmax,
            AppliedTerminalPeel::LogSoftmax,
        ] {
            let (text, refused) = verification_result_text(&result, peel);
            assert!(refused);
            assert!(text.contains("Status: UNKNOWN"));
            assert!(text.contains("preactivation"));
            assert!(text.contains("not original-model Y"));
            assert!(
                !text.contains("0.125"),
                "counterexample input leaked: {text}"
            );
            assert!(!text.contains("42.25"), "raw peeled output leaked: {text}");

            let (json, refused) = verification_json_value(
                &result,
                PropagationMethod::Crown,
                &cpu_backend_receipt(),
                0.0,
                None,
                None,
                peel,
            )
            .expect("Softmax-family result JSON");
            assert!(refused);
            assert_eq!(json["status"], "unknown");
            assert!(json.get("counterexample").is_none());
            assert!(json.get("output").is_none());
            assert!(json.get("violation").is_none());
            assert!(json.get("explanation").is_none());
            assert_eq!(json["output_coordinates"], "peeled_preactivation");
            assert_eq!(
                compute_publication_exit_code(&result, None, true, refused),
                exit_codes::UNKNOWN,
                "--allow-unknown must not hide a refused witness"
            );
        }
    }

    /// Helper: build a VnnLibSpec with given num_outputs and constraints.
    fn spec_with_constraints(num_outputs: usize, constraints: Vec<OutputConstraint>) -> VnnLibSpec {
        VnnLibSpec {
            num_inputs: 1,
            num_outputs,
            input_bounds: vec![(0.0, 1.0)],
            output_constraints: constraints,
            output_constraint_clauses: Vec::new(),
            is_disjunction: false,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        }
    }

    #[test]
    fn test_evaluate_property_status_valid_indices() {
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(2.0, 3.0)];
        let result = verified_with_bounds(bounds);
        // Y_0 <= Y_1: lower(0)=0.0 <= upper(1)=3.0 → satisfiable
        let spec = spec_with_constraints(2, vec![OutputConstraint::LessEq(0, 1)]);
        let status = evaluate_property_status(&result, Some(&spec)).unwrap();
        assert_eq!(status, Some("unknown"));
    }

    #[test]
    fn test_evaluate_property_status_safe() {
        let bounds = vec![Bound::new(5.0, 6.0), Bound::new(0.0, 1.0)];
        let result = verified_with_bounds(bounds);
        // Y_0 <= Y_1: lower(0)=5.0 <= upper(1)=1.0 → false → not satisfiable → safe
        let spec = spec_with_constraints(2, vec![OutputConstraint::LessEq(0, 1)]);
        let status = evaluate_property_status(&result, Some(&spec)).unwrap();
        assert_eq!(status, Some("safe"));
    }

    #[test]
    fn test_evaluate_property_status_oob_index_returns_error() {
        // Regression test for #2878: out-of-bounds constraint index must not panic.
        let bounds = vec![Bound::new(0.0, 1.0)]; // only 1 output
        let result = verified_with_bounds(bounds);
        // Constraint references index 5 — out of bounds
        let spec = spec_with_constraints(10, vec![OutputConstraint::LessEqConst(5, 0.5)]);
        let err = evaluate_property_status(&result, Some(&spec));
        assert!(err.is_err(), "Expected error for OOB index, got {:?}", err);
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("output index 5") && msg.contains("1 outputs"),
            "Error message should mention the index and bounds length: {msg}"
        );
    }

    #[test]
    fn test_evaluate_property_status_oob_relational_returns_error() {
        // Regression test for #2878: relational constraint with OOB index j.
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(2.0, 3.0)]; // 2 outputs
        let result = verified_with_bounds(bounds);
        // Y_0 >= Y_99: index 99 is out of bounds
        let spec = spec_with_constraints(100, vec![OutputConstraint::GreaterEq(0, 99)]);
        let err = evaluate_property_status(&result, Some(&spec));
        assert!(err.is_err(), "Expected error for OOB relational index");
    }

    #[test]
    fn test_evaluate_property_status_no_spec() {
        let bounds = vec![Bound::new(0.0, 1.0)];
        let result = verified_with_bounds(bounds);
        let status = evaluate_property_status(&result, None).unwrap();
        assert_eq!(status, None);
    }

    #[test]
    fn test_evaluate_property_status_no_constraints() {
        let bounds = vec![Bound::new(0.0, 1.0)];
        let result = verified_with_bounds(bounds);
        let spec = spec_with_constraints(1, vec![]);
        let status = evaluate_property_status(&result, Some(&spec)).unwrap();
        assert_eq!(status, None);
    }

    #[test]
    fn test_evaluate_property_status_nonfinite_bounds_and_constants_are_unknown() {
        let result = verified_with_bounds(vec![Bound::new_allow_infinite(
            f32::INFINITY,
            f32::INFINITY,
        )]);
        let spec = spec_with_constraints(1, vec![OutputConstraint::LessEqConst(0, 0.0)]);
        assert_eq!(
            evaluate_property_status(&result, Some(&spec)).unwrap(),
            Some("unknown")
        );

        let result = verified_with_bounds(vec![Bound::new(1.0, 2.0)]);
        for constant in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let spec = spec_with_constraints(1, vec![OutputConstraint::LessEqConst(0, constant)]);
            assert_eq!(
                evaluate_property_status(&result, Some(&spec)).unwrap(),
                Some("unknown")
            );
        }

        let result = verified_with_bounds(vec![
            Bound::new(1.0, 2.0),
            Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
        ]);
        let spec = spec_with_constraints(2, vec![OutputConstraint::LessEqConst(0, 0.0)]);
        assert_eq!(
            evaluate_property_status(&result, Some(&spec)).unwrap(),
            Some("safe"),
            "an unreferenced conservative output row must not reduce completeness"
        );
    }

    #[test]
    fn test_evaluate_property_status_compares_exact_f64_constants() {
        let result = verified_with_bounds(vec![Bound::new(1.0, 1.0)]);
        let just_below_one = f64::from_bits(1.0_f64.to_bits() - 1);
        let just_above_one = f64::from_bits(1.0_f64.to_bits() + 1);

        let below =
            spec_with_constraints(1, vec![OutputConstraint::LessEqConst(0, just_below_one)]);
        assert_eq!(
            evaluate_property_status(&result, Some(&below)).unwrap(),
            Some("safe")
        );

        let above =
            spec_with_constraints(1, vec![OutputConstraint::GreaterEqConst(0, just_above_one)]);
        assert_eq!(
            evaluate_property_status(&result, Some(&above)).unwrap(),
            Some("safe")
        );
    }

    #[test]
    fn test_evaluate_property_status_honors_clause_aggregation_mode() {
        let result = verified_with_bounds(vec![Bound::new(1.0, 2.0)]);
        let mut spec = spec_with_constraints(1, Vec::new());
        spec.output_constraint_clauses = vec![
            vec![OutputConstraint::LessEqConst(0, 0.0)], // impossible
            vec![OutputConstraint::GreaterEqConst(0, 0.0)], // possible
        ];

        spec.is_disjunction = true;
        assert_eq!(
            evaluate_property_status(&result, Some(&spec)).unwrap(),
            Some("unknown"),
            "OR-unsafe remains possible when any clause may hold"
        );

        spec.is_disjunction = false;
        assert_eq!(
            evaluate_property_status(&result, Some(&spec)).unwrap(),
            Some("safe"),
            "AND-unsafe is impossible when any clause is impossible"
        );
    }

    // --- Regression tests for #3295: status override from property_status ---

    /// When VerificationResult is Verified but property is unknown, exit code must be UNKNOWN.
    /// Regression test for #3295: verify command used to always report "verified" status.
    #[test]
    fn test_exit_code_verified_result_but_unknown_property() {
        let bounds = vec![Bound::new(0.0, 1.0)];
        let result = verified_with_bounds(bounds);
        let code = compute_exit_code(&result, Some("unknown"), false);
        assert_eq!(
            code,
            exit_codes::UNKNOWN,
            "Verified result with unknown property_status must exit UNKNOWN"
        );
    }

    #[test]
    fn test_allow_unknown_never_masks_timeout() {
        let result = VerificationResult::Timeout {
            provenance: SoundnessProvenance::sound(),
            partial_bounds: None,
            actual_method: None,
        };
        assert_eq!(
            compute_exit_code(&result, None, true),
            exit_codes::TIMEOUT,
            "--allow-unknown must leave timeout at exit code 3"
        );
    }

    /// When VerificationResult is Verified and property is safe, exit code is VERIFIED.
    #[test]
    fn test_exit_code_verified_result_safe_property() {
        let bounds = vec![Bound::new(5.0, 6.0)];
        let result = verified_with_bounds(bounds);
        let code = compute_exit_code(&result, Some("safe"), false);
        assert_eq!(code, exit_codes::VERIFIED);
    }

    /// When VerificationResult is Verified but no property_status, exit code defaults to VERIFIED.
    #[test]
    fn test_exit_code_verified_result_no_property() {
        let bounds = vec![Bound::new(0.0, 1.0)];
        let result = verified_with_bounds(bounds);
        let code = compute_exit_code(&result, None, false);
        assert_eq!(code, exit_codes::VERIFIED);
    }

    /// Regression test for #3133: json_f32 must render non-finite f32 as strings,
    /// not null.
    #[test]
    fn test_json_f32_non_finite_renders_as_string_3133() {
        // Finite values produce numbers
        let v = json_f32(1.5);
        assert!(v.is_number(), "finite f32 should be a JSON number: {v}");

        let v = json_f32(0.0);
        assert!(v.is_number(), "zero should be a JSON number: {v}");

        let v = json_f32(-2.5);
        assert!(
            v.is_number(),
            "negative finite should be a JSON number: {v}"
        );

        // Infinity renders as string "Infinity"
        let v = json_f32(f32::INFINITY);
        assert_eq!(v, serde_json::Value::String("Infinity".to_string()));

        // Negative infinity renders as string "-Infinity"
        let v = json_f32(f32::NEG_INFINITY);
        assert_eq!(v, serde_json::Value::String("-Infinity".to_string()));

        // NaN renders as string "NaN"
        let v = json_f32(f32::NAN);
        assert_eq!(v, serde_json::Value::String("NaN".to_string()));

        // Full round-trip: json!() embedding should preserve strings
        let obj = serde_json::json!({
            "lower": json_f32(f32::NEG_INFINITY),
            "upper": json_f32(f32::INFINITY)
        });
        assert_eq!(obj["lower"], "-Infinity");
        assert_eq!(obj["upper"], "Infinity");
        // Must NOT be null (the old behavior)
        assert!(
            !obj["lower"].is_null(),
            "Inf bounds must not be null (#3133)"
        );
        assert!(
            !obj["upper"].is_null(),
            "Inf bounds must not be null (#3133)"
        );
    }

    #[test]
    fn test_actual_method_matches_requested_typed() {
        // Matching propagation methods
        assert!(actual_method_matches_requested(
            &MethodUsed::Crown,
            PropagationMethod::Crown
        ));
        assert!(actual_method_matches_requested(
            &MethodUsed::BetaCrown,
            PropagationMethod::BetaCrown
        ));
        assert!(actual_method_matches_requested(
            &MethodUsed::Ibp,
            PropagationMethod::Ibp
        ));
        // Degraded fallback: requested AlphaCrown but got Crown
        assert!(!actual_method_matches_requested(
            &MethodUsed::Crown,
            PropagationMethod::AlphaCrown
        ));
        // Non-propagation methods never match a propagation request
        assert!(!actual_method_matches_requested(
            &MethodUsed::SmtRefiner,
            PropagationMethod::BetaCrown
        ));
        assert!(!actual_method_matches_requested(
            &MethodUsed::MipHiGHS,
            PropagationMethod::Crown
        ));
    }
}
