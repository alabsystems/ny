// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX-backed model comparison commands.

use super::backend::{resolve_gemm_backend_with_factory, GemmBackendResolution};
use super::verify::json_f32;
use crate::commands::inspect_model::{render_inspect_output, InspectOutput};
use crate::BackendArg;
use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use ny_core::nan_propagating_max;
use ny_gpu::ComputeDevice;
use ny_onnx::load_onnx;
use ny_propagate::{Network, PropagationMethod};
use ny_tensor::BoundedTensor;
use serde_json::json;
use std::path::Path;
use tracing::info;

/// Handle the `ny inspect` command.
pub(crate) fn handle_inspect_command(
    model: &Path,
    native: bool,
    cost: bool,
    timing_profile: Option<&Path>,
    json: bool,
) -> Result<()> {
    match render_inspect_output(model, native, cost, json, timing_profile)? {
        InspectOutput::Text(output) => println!("{output}"),
        InspectOutput::Json(output) => println!("{}", serde_json::to_string_pretty(&output)?),
    }
    Ok(())
}

/// Handle the `ny compare` command.
///
/// Compares bound propagation results between two single-input ONNX models.
/// Reports whether bounds match within the specified tolerance, overlap
/// statistics, and detailed violation information.
// Justification: Compare command needs both model paths, tolerance, perturbation,
// method selection, backend choice, and output flags — all from CLI arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_compare_command(
    reference: &Path,
    target: &Path,
    tolerance: f32,
    epsilon: f32,
    method: &str,
    backend: BackendArg,
    gpu: bool,
    verbose: bool,
    json: bool,
) -> Result<()> {
    if !tolerance.is_finite() {
        anyhow::bail!("Tolerance must be finite (got {tolerance})");
    }
    if tolerance < 0.0 {
        anyhow::bail!("Tolerance must be non-negative (got {tolerance})");
    }
    if !epsilon.is_finite() {
        anyhow::bail!("Epsilon must be finite (got {epsilon})");
    }
    if epsilon < 0.0 {
        anyhow::bail!("Epsilon must be non-negative (got {epsilon})");
    }

    let requested_method = parse_compare_method(method)?;
    let method_label = compare_method_label(requested_method)?;
    let backend_resolution = resolve_compare_backend(backend, gpu, json, requested_method);
    let gemm_engine = backend_resolution.gemm_engine();

    info!("Comparing models:");
    info!("  Reference: {}", reference.display());
    info!("  Target: {}", target.display());
    info!("  Tolerance: {}", tolerance);
    info!("  Epsilon: {}", epsilon);
    info!("  Method: {}", method_label);
    info!("  Backend: {}", backend_resolution.backend);

    // Load both models
    let ref_model = load_onnx(reference)
        .with_context(|| format!("Failed to load reference model: {}", reference.display()))?;
    let target_model = load_onnx(target)
        .with_context(|| format!("Failed to load target model: {}", target.display()))?;

    // Convert to propagation networks
    let ref_network = ref_model
        .to_propagate_network()
        .context("Failed to convert reference model to propagation network")?;
    let target_network = target_model
        .to_propagate_network()
        .context("Failed to convert target model to propagation network")?;

    // Get input shapes (use reference model's input shape)
    let ref_onnx_net = &ref_model.network;
    let target_onnx_net = &target_model.network;

    if ref_onnx_net.inputs.len() != 1 || target_onnx_net.inputs.len() != 1 {
        anyhow::bail!(
            "Compare supports single-input models only (reference inputs: {}, target inputs: {})",
            ref_onnx_net.inputs.len(),
            target_onnx_net.inputs.len()
        );
    }

    if !json {
        println!("Reference model: {}", ref_onnx_net.name);
        println!(
            "  Inputs: {:?}",
            ref_onnx_net
                .inputs
                .iter()
                .map(|i| (&i.name, &i.shape))
                .collect::<Vec<_>>()
        );
        println!(
            "  Outputs: {:?}",
            ref_onnx_net
                .outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        );
        println!("  Layers: {}", ref_network.layers().len());

        println!("Target model: {}", target_onnx_net.name);
        println!(
            "  Inputs: {:?}",
            target_onnx_net
                .inputs
                .iter()
                .map(|i| (&i.name, &i.shape))
                .collect::<Vec<_>>()
        );
        println!(
            "  Outputs: {:?}",
            target_onnx_net
                .outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        );
        println!("  Layers: {}", target_network.layers().len());
    }

    // Verify input shapes match
    let ref_input_shape: Vec<usize> = ref_onnx_net.inputs[0]
        .shape
        .iter()
        .map(|&d| d.max(1) as usize)
        .collect();

    let target_input_shape: Vec<usize> = target_onnx_net.inputs[0]
        .shape
        .iter()
        .map(|&d| d.max(1) as usize)
        .collect();

    if ref_input_shape != target_input_shape {
        anyhow::bail!(
            "Input shapes don't match: reference {:?} vs target {:?}",
            ref_input_shape,
            target_input_shape
        );
    }

    // Create bounded input
    let input_data = ArrayD::from_elem(IxDyn(&ref_input_shape), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, epsilon)?;

    if !json {
        println!("\nInput: shape {:?}, epsilon {}", ref_input_shape, epsilon);
        println!("Method: {}", method_label);
    }

    let propagate = |network: &Network| -> Result<BoundedTensor> {
        match requested_method {
            PropagationMethod::Ibp => Ok(network.propagate_ibp(&input)?),
            PropagationMethod::Crown => {
                Ok(network.propagate_crown_with_engine(&input, gemm_engine)?)
            }
            PropagationMethod::AlphaCrown => {
                Ok(network.propagate_alpha_crown_with_engine(&input, gemm_engine)?)
            }
            _ => anyhow::bail!("Unsupported method for inspect compare: {}", method_label),
        }
    };

    // Run bound propagation on both models
    let start = std::time::Instant::now();
    let ref_output = propagate(&ref_network)?;
    let ref_time = start.elapsed();

    let start = std::time::Instant::now();
    let target_output = propagate(&target_network)?;
    let target_time = start.elapsed();

    // Compare outputs
    if !json {
        println!("\n--- Propagation Results ---");
        println!(
            "Reference: {:?} in {:.2}ms",
            ref_output.shape(),
            ref_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Lower: min={:.6}, max={:.6}",
            ref_output
                .lower()
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            ref_output
                .lower()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Upper: min={:.6}, max={:.6}",
            ref_output
                .upper()
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            ref_output
                .upper()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!("  Max width: {:.6e}", ref_output.max_width());

        println!(
            "Target: {:?} in {:.2}ms",
            target_output.shape(),
            target_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Lower: min={:.6}, max={:.6}",
            target_output
                .lower()
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            target_output
                .lower()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Upper: min={:.6}, max={:.6}",
            target_output
                .upper()
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            target_output
                .upper()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!("  Max width: {:.6e}", target_output.max_width());
    }

    // Verify shapes match
    if ref_output.shape() != target_output.shape() {
        if !json {
            println!("\n--- FAIL: Output shapes don't match ---");
        }
        anyhow::bail!(
            "Output shapes don't match: reference {:?} vs target {:?}",
            ref_output.shape(),
            target_output.shape()
        );
    }

    // Compare element-wise: check if bounds are equivalent within tolerance
    // Two models are equivalent if for all elements:
    //   |ref_lower - target_lower| <= tolerance AND |ref_upper - target_upper| <= tolerance
    let ref_lower = ref_output.lower();
    let ref_upper = ref_output.upper();
    let target_lower = target_output.lower();
    let target_upper = target_output.upper();

    let mut max_lower_diff: f32 = 0.0;
    let mut max_upper_diff: f32 = 0.0;
    let mut violations = Vec::new();

    for (idx, (((&rl, &ru), &tl), &tu)) in ref_lower
        .iter()
        .zip(ref_upper.iter())
        .zip(target_lower.iter())
        .zip(target_upper.iter())
        .enumerate()
    {
        let lower_diff: f32 = (rl - tl).abs();
        let upper_diff: f32 = (ru - tu).abs();
        // nan_propagating_max propagates NaN instead of absorbing it (#2852).
        max_lower_diff = nan_propagating_max(max_lower_diff, lower_diff);
        max_upper_diff = nan_propagating_max(max_upper_diff, upper_diff);
        let exceeds = lower_diff > tolerance
            || upper_diff > tolerance
            || !lower_diff.is_finite()
            || !upper_diff.is_finite();
        if exceeds && (violations.len() < 10 || verbose) {
            violations.push((idx, rl, ru, tl, tu, lower_diff, upper_diff));
        }
    }

    // Overlap metric: non-finite endpoints conservatively count as non-overlapping (#2852).
    let total = ref_lower.len();
    let overlap_count = ref_lower
        .iter()
        .zip(ref_upper.iter())
        .zip(target_lower.iter())
        .zip(target_upper.iter())
        .filter(|(((&rl, &ru), &tl), &tu)| {
            rl.is_finite()
                && ru.is_finite()
                && tl.is_finite()
                && tu.is_finite()
                && rl.max(tl) <= ru.min(tu)
        })
        .count();
    let overlap_pct = if total == 0 {
        0.0
    } else {
        100.0 * overlap_count as f64 / total as f64
    };

    let equivalent = max_lower_diff <= tolerance && max_upper_diff <= tolerance; // NaN → non-equivalent

    if json {
        let summary = json!({
            "equivalent": equivalent,
            "method": method_label,
            "max_lower_diff": json_f32(max_lower_diff),
            "max_upper_diff": json_f32(max_upper_diff),
            "tolerance": json_f32(tolerance),
            "overlap_pct": overlap_pct,
            "ref_max_width": json_f32(ref_output.max_width()),
            "target_max_width": json_f32(target_output.max_width()),
            "output_shape": ref_output.shape(),
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("\n--- Comparison Results ---");
        println!("Max lower bound diff: {:.6e}", max_lower_diff);
        println!("Max upper bound diff: {:.6e}", max_upper_diff);
        println!("Tolerance: {:.6e}", tolerance);
        println!(
            "Bound overlap: {}/{} ({:.2}%)",
            overlap_count, total, overlap_pct
        );

        if equivalent {
            println!("\n✓ EQUIVALENT: Models produce matching bounds within tolerance");
        } else {
            println!("\n✗ NOT EQUIVALENT: Models differ beyond tolerance");
            println!(
                "\nViolations (first {}{}): ",
                violations.len(),
                if violations.len() < 10 && !verbose {
                    ""
                } else {
                    ", use --verbose for all"
                }
            );
            for (idx, rl, ru, tl, tu, ld, ud) in &violations {
                println!(
                    "  [{}] ref=[{:.6}, {:.6}] target=[{:.6}, {:.6}] diff=({:.3e}, {:.3e})",
                    idx, rl, ru, tl, tu, ld, ud
                );
            }
        }
    }

    Ok(())
}

fn parse_compare_method(method: &str) -> Result<PropagationMethod> {
    let method = method.trim().to_ascii_lowercase();
    match method.as_str() {
        "ibp" => Ok(PropagationMethod::Ibp),
        "crown" => Ok(PropagationMethod::Crown),
        "alpha" | "alpha-crown" | "alpha_crown" => Ok(PropagationMethod::AlphaCrown),
        _ => anyhow::bail!(
            "Unknown method: {}. Use ibp, crown, or alpha (aliases: alpha-crown, alpha_crown)",
            method
        ),
    }
}

fn compare_method_label(method: PropagationMethod) -> Result<&'static str> {
    match method {
        PropagationMethod::Ibp => Ok("ibp"),
        PropagationMethod::Crown => Ok("crown"),
        PropagationMethod::AlphaCrown => Ok("alpha"),
        _ => anyhow::bail!(
            "Unsupported compare method: {:?}. Use ibp, crown, or alpha.",
            method
        ),
    }
}

fn compare_method_uses_gemm_engine(method: PropagationMethod) -> bool {
    matches!(
        method,
        PropagationMethod::Crown | PropagationMethod::AlphaCrown
    )
}

fn resolve_compare_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    method: PropagationMethod,
) -> GemmBackendResolution<ComputeDevice> {
    resolve_compare_backend_with_factory(backend, gpu, json, method, |effective_backend| {
        Ok(ComputeDevice::new(effective_backend.into())?)
    })
}

pub(super) fn resolve_compare_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    method: PropagationMethod,
    build_device: F,
) -> GemmBackendResolution<T>
where
    F: FnOnce(BackendArg) -> Result<T>,
{
    if compare_method_uses_gemm_engine(method) {
        resolve_gemm_backend_with_factory(backend, gpu, json, build_device)
    } else {
        GemmBackendResolution::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::{compare_method_label, compare_method_uses_gemm_engine, parse_compare_method};
    use ny_propagate::PropagationMethod;

    #[test]
    fn parse_compare_method_accepts_aliases() {
        assert_eq!(parse_compare_method("ibp").unwrap(), PropagationMethod::Ibp);
        assert_eq!(
            parse_compare_method("crown").unwrap(),
            PropagationMethod::Crown
        );
        assert_eq!(
            parse_compare_method("alpha").unwrap(),
            PropagationMethod::AlphaCrown
        );
        assert_eq!(
            parse_compare_method("alpha-crown").unwrap(),
            PropagationMethod::AlphaCrown
        );
        assert_eq!(
            parse_compare_method("alpha_crown").unwrap(),
            PropagationMethod::AlphaCrown
        );
    }

    #[test]
    fn parse_compare_method_rejects_unknown() {
        let err = parse_compare_method("beta").unwrap_err().to_string();
        assert!(err.contains("Unknown method"));
    }

    #[test]
    fn compare_method_label_rejects_unsupported() {
        let err = compare_method_label(PropagationMethod::SdpCrown)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Unsupported compare method"));
        let err = compare_method_label(PropagationMethod::BetaCrown)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Unsupported compare method"));
    }

    #[test]
    fn compare_method_engine_usage_matches_supported_methods() {
        assert!(!compare_method_uses_gemm_engine(PropagationMethod::Ibp));
        assert!(compare_method_uses_gemm_engine(PropagationMethod::Crown));
        assert!(compare_method_uses_gemm_engine(
            PropagationMethod::AlphaCrown
        ));
    }
}
