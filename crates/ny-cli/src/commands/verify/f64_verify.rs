// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision (f64) verification path for soundnessbench/sat_relu.
//!
//! When `--double-fp` is active, this module converts the sequential network
//! to f64 types and runs propagation entirely in f64. The f64 output bounds
//! are converted back to f32 with directed rounding (sound) before rendering.
//!
//! Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).
//! Design: `designs/2026-03-04-f64-propagation-path.md` step 6.

use anyhow::Result;
use ny_core::{Bound, SoundnessProvenance, VerificationResult, VerificationSpec};
use ny_propagate::{
    convert_network_to_f64, propagate_network_f64, F64PropagationMode, PropagationMethod,
};
use ny_tensor::BoundedTensor64;
use std::path::Path;
use tracing::info;

use super::model_load::VerifiableNetwork;

/// Run verification using the f64 propagation path.
///
/// Only supports sequential networks with Linear+Conv2D+ReLU+Flatten layers.
/// Graph networks (DAG) are not supported in f64 mode.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_f64_verification(
    network: &VerifiableNetwork,
    spec: &VerificationSpec,
    requested_method: PropagationMethod,
    effective_method: PropagationMethod,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
    property: Option<&Path>,
    strict: bool,
    allow_unknown: bool,
    json: bool,
) -> Result<()> {
    let sequential = match network {
        VerifiableNetwork::Sequential(net) => net,
        VerifiableNetwork::Graph(_) => {
            anyhow::bail!(
                "--double-fp only supports sequential networks (Linear+Conv2D+ReLU). \
                 Graph/DAG networks are not supported in f64 mode."
            );
        }
    };

    if !json {
        info!("Using f64 (double precision) propagation path");
    }

    // Convert Network layers to f64 (returns Err for unsupported layer types)
    let layers_f64 = convert_network_to_f64(sequential.layers())?;

    // Convert input bounds to f64
    let input_f32 =
        ny_propagate::Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;
    let input_f64 = BoundedTensor64::from_f32(&input_f32);

    // Select f64 propagation mode
    let f64_mode = match effective_method {
        PropagationMethod::Ibp => F64PropagationMode::Ibp,
        // SDP-CROWN is only valid over an ℓ2 input ball, and this path (like the
        // f32 verifiers, which also refuse it) only carries ℓ∞ box specs. Routing
        // it to f64 CROWN would be sound over the box but a silent method
        // substitution, so refuse with the same message as the f32 path.
        PropagationMethod::SdpCrown => {
            anyhow::bail!(
                "SDP-CROWN requires an ℓ2 input ball, but the specification declares an \
                 ℓ∞ input box; use CROWN or α-CROWN instead"
            );
        }
        // Remaining CROWN variants (Crown, AlphaCrown, BetaCrown) use the f64 CROWN
        // path. f64 CROWN uses fixed-slope relaxation (no alpha optimization in f64 yet).
        _ => F64PropagationMode::Crown,
    };

    if !json {
        info!(
            "f64 mode: {:?}, {} layers converted",
            f64_mode,
            layers_f64.len()
        );
    }

    // Run f64 propagation
    let output_f64 = propagate_network_f64(&layers_f64, &input_f64, f64_mode)?;

    // Convert back to f32 with directed rounding (sound: lower rounds down, upper rounds up)
    let output_f32 = output_f64.to_f32_sound();

    // Build output bounds as Bound vec for spec checking. A non-finite or inverted
    // interval must refuse a verdict rather than panic inside `Bound`.
    let output_bounds: Vec<Bound> = output_f32
        .lower()
        .iter()
        .zip(output_f32.upper().iter())
        .enumerate()
        .map(|(i, (&l, &u))| {
            Bound::try_new_allow_infinite(l, u).map_err(|e| {
                anyhow::anyhow!("f64 output bound Y_{i} is invalid (lower={l}, upper={u}): {e}")
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let provenance = SoundnessProvenance::default(); // f64 is sound by design
    let actual_method_tag = match f64_mode {
        F64PropagationMode::Ibp => ny_core::MethodUsed::IbpF64,
        F64PropagationMode::Crown => ny_core::MethodUsed::CrownF64,
    };

    // Check strict mode: fail if method fallback occurred
    if strict
        && !super::result::actual_method_matches_requested(&actual_method_tag, requested_method)
    {
        anyhow::bail!(
            "Strict mode: requested method '{:?}' but fell back to '{}'. \
            Remove --strict to allow fallback.",
            requested_method,
            actual_method_tag.as_str()
        );
    }

    // Check spec satisfaction
    let result = check_spec_f64(
        &output_bounds,
        spec.output_bounds(),
        &actual_method_tag,
        provenance,
    )?;

    // Check property constraints if VNN-LIB spec is provided
    let property_status = super::result::evaluate_property_status(&result, vnnlib_spec)?;

    // Render output
    if json {
        render_f64_json(
            &result,
            &actual_method_tag,
            &output_bounds,
            property_status,
            property,
        )?;
    } else {
        render_f64_text(&result, &output_bounds, property_status)?;
    }

    // Determine and apply exit code
    let exit_code = super::result::compute_exit_code(&result, property_status, allow_unknown);
    if exit_code != super::result::exit_codes::VERIFIED {
        std::process::exit(exit_code);
    }

    Ok(())
}

/// Check if f64 output bounds satisfy the specification.
///
/// `Verified` here means only that propagation landed inside the spec's output box,
/// which for a CLI-built spec is the vacuous ±infinity box — it does NOT mean a
/// property was proven. The VNN-LIB property is the real gate and is evaluated by
/// the caller, exactly as on the standard path.
fn check_spec_f64(
    output_bounds: &[Bound],
    spec_bounds: &[Bound],
    actual_method: &ny_core::MethodUsed,
    provenance: SoundnessProvenance,
) -> Result<VerificationResult> {
    // Zipping a shorter spec would leave the trailing outputs unchecked.
    if spec_bounds.len() != output_bounds.len() {
        anyhow::bail!(
            "f64 propagation produced {} output bounds but the specification constrains {}",
            output_bounds.len(),
            spec_bounds.len()
        );
    }

    // Check each output against its specification
    let mut all_verified = true;
    let mut max_gap: f32 = 0.0;

    for (computed, required) in output_bounds.iter().zip(spec_bounds.iter()) {
        let lower_gap = if computed.lower() < required.lower() {
            required.lower() - computed.lower()
        } else {
            0.0
        };
        let upper_gap = if computed.upper() > required.upper() {
            computed.upper() - required.upper()
        } else {
            0.0
        };
        let gap = lower_gap.max(upper_gap);
        if gap > 0.0 {
            all_verified = false;
            max_gap = max_gap.max(gap);
        }
    }

    Ok(if all_verified {
        VerificationResult::Verified {
            provenance,
            output_bounds: output_bounds.to_vec(),
            proof: None,
            actual_method: Some(actual_method.clone()),
        }
    } else {
        VerificationResult::Unknown {
            provenance,
            bounds: output_bounds.to_vec(),
            reason: ny_core::UnknownReason::BoundsTooLoose { gap: Some(max_gap) },
            actual_method: Some(actual_method.clone()),
        }
    })
}

/// The verdict actually reported, after the VNN-LIB property gate.
///
/// A `Verified` from `check_spec_f64` only records that propagation completed; when a
/// property was supplied it is the property status that decides. Mirrors the override
/// in `result::render_json` and the mapping in `result::compute_exit_code`, so text,
/// JSON, and exit code cannot disagree.
fn reported_status(result: &VerificationResult, property_status: Option<&str>) -> &'static str {
    match result {
        VerificationResult::Verified { .. } => match property_status {
            Some("safe") | None => "verified",
            Some(_) => "unknown",
        },
        VerificationResult::Violated { .. } => "violated",
        VerificationResult::Unknown { .. } => "unknown",
        VerificationResult::Timeout { .. } => "timeout",
    }
}

/// Render f64 verification result as JSON.
fn render_f64_json(
    result: &VerificationResult,
    actual_method: &ny_core::MethodUsed,
    output_bounds: &[Bound],
    property_status: Option<&str>,
    property: Option<&Path>,
) -> Result<()> {
    use super::result::json_f32;

    let bounds_arr: Vec<serde_json::Value> = output_bounds
        .iter()
        .map(|b| {
            serde_json::json!({
                "lower": json_f32(b.lower()),
                "upper": json_f32(b.upper()),
            })
        })
        .collect();

    let mut output = serde_json::json!({
        "status": reported_status(result, property_status),
        "method": actual_method.as_str(),
        "double_fp": true,
        "output_bounds": bounds_arr,
    });

    if let Some(status) = property_status {
        output["property_status"] = serde_json::json!(status);
    }
    if let Some(p) = property {
        output["property_file"] = serde_json::json!(p.display().to_string());
    }
    output["soundness"] = serde_json::to_value(result.provenance())?;

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Render f64 verification result as text.
fn render_f64_text(
    result: &VerificationResult,
    output_bounds: &[Bound],
    property_status: Option<&str>,
) -> Result<()> {
    println!(
        "Result: {} (f64 double precision)",
        reported_status(result, property_status).to_uppercase()
    );
    if let VerificationResult::Unknown { reason, .. } = result {
        println!("  Reason: {reason:?}");
    }

    if !output_bounds.is_empty() {
        println!(
            "  Output bounds: [{:.6}, {:.6}]",
            output_bounds[0].lower(),
            output_bounds[0].upper()
        );
        if output_bounds.len() > 1 {
            println!("  ({} total output dimensions)", output_bounds.len());
        }
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
