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
use crate::BackendArg;

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
    effective_backend: BackendArg,
    epsilon: f32,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
    property: Option<&Path>,
    strict: bool,
    require_sound: bool,
    allow_unknown: bool,
    json: bool,
) -> Result<()> {
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
    if json {
        render_json(
            &result,
            requested_method,
            effective_backend,
            epsilon,
            property_status,
            property,
        )?;
    } else {
        render_text(&result, property_status)?;
    }

    // Determine and apply exit code
    let exit_code = compute_exit_code(&result, property_status, allow_unknown);
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

    fn get_bound(bounds: &[Bound], idx: usize) -> Result<&Bound> {
        bounds.get(idx).ok_or_else(|| {
            anyhow::anyhow!(
                "VNN-LIB constraint references output index {} but model has {} outputs",
                idx,
                bounds.len()
            )
        })
    }

    let check_constraint_satisfiable =
        |bounds: &[Bound], constraint: &OutputConstraint| -> Result<bool> {
            use ny_tensor::{next_down_f32, next_up_f32};
            Ok(match constraint {
                OutputConstraint::LessEq(i, j) => {
                    get_bound(bounds, *i)?.lower() <= get_bound(bounds, *j)?.upper()
                }
                OutputConstraint::GreaterEq(i, j) => {
                    get_bound(bounds, *i)?.upper() >= get_bound(bounds, *j)?.lower()
                }
                OutputConstraint::LessThan(i, j) => {
                    get_bound(bounds, *i)?.lower() < get_bound(bounds, *j)?.upper()
                }
                OutputConstraint::GreaterThan(i, j) => {
                    get_bound(bounds, *i)?.upper() > get_bound(bounds, *j)?.lower()
                }
                // Directed rounding for f64→f32 constant conversion: round in the
                // direction that makes satisfiability *easier* to achieve, so we never
                // falsely declare a clause unsatisfiable (which would incorrectly
                // produce a "safe" verdict). See constraint_plan.rs #2658.
                OutputConstraint::LessEqConst(i, c) => {
                    // lower(i) <= c: round c UP so the check is conservative.
                    get_bound(bounds, *i)?.lower() <= next_up_f32(*c as f32)
                }
                OutputConstraint::GreaterEqConst(i, c) => {
                    // upper(i) >= c: round c DOWN so the check is conservative.
                    get_bound(bounds, *i)?.upper() >= next_down_f32(*c as f32)
                }
                OutputConstraint::LessThanConst(i, c) => {
                    // lower(i) < c: round c UP so the check is conservative.
                    get_bound(bounds, *i)?.lower() < next_up_f32(*c as f32)
                }
                OutputConstraint::GreaterThanConst(i, c) => {
                    // upper(i) > c: round c DOWN so the check is conservative.
                    get_bound(bounds, *i)?.upper() > next_down_f32(*c as f32)
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
                // Check each clause; propagate index-out-of-bounds errors.
                let mut any_satisfiable = false;
                for clause in &clauses {
                    let mut all_satisfied = true;
                    for c in *clause {
                        if !check_constraint_satisfiable(output_bounds, c)? {
                            all_satisfied = false;
                            break;
                        }
                    }
                    if all_satisfied {
                        any_satisfiable = true;
                        break;
                    }
                }

                if any_satisfiable {
                    Ok(Some("unknown"))
                } else {
                    Ok(Some("safe"))
                }
            }
        }
        _ => Ok(None),
    }
}

/// Render verification result as JSON.
fn render_json(
    result: &ny_core::VerificationResult,
    requested_method: PropagationMethod,
    effective_backend: BackendArg,
    epsilon: f32,
    property_status: Option<&str>,
    property: Option<&Path>,
) -> Result<()> {
    use serde_json::json;
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
                "backend": effective_backend.to_string()
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
            let mut json_val = json!({
                "status": "violated",
                "counterexample": counterexample,
                "output": output,
                "epsilon": epsilon,
                "method": method_str,
                "backend": effective_backend.to_string()
            });
            if let Some(am) = actual_method {
                if let Some(obj) = json_val.as_object_mut() {
                    obj.insert("actual_method".to_string(), json!(am));
                }
            }
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
                "backend": effective_backend.to_string()
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
                "backend": effective_backend.to_string()
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

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Render verification result as human-readable text.
fn render_text(result: &ny_core::VerificationResult, property_status: Option<&str>) -> Result<()> {
    println!("\nVerification Result:");
    println!("{:?}", result);

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
    Ok(())
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
        ny_core::VerificationResult::Timeout { .. } => {
            if allow_unknown {
                exit_codes::VERIFIED
            } else {
                exit_codes::TIMEOUT
            }
        }
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
