// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX-backed model analysis commands: diff, sensitivity, quantization, and profiling.
//!
//! Provides CLI handlers for:
//! - `ny diff` - Layer-by-layer comparison between two ONNX models
//! - `ny sensitivity` - Bound sensitivity analysis
//! - `ny quantize-check` - Quantization safety analysis for an ONNX model
//! - `ny profile-bounds` - Bound propagation profiling

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

fn validate_nonnegative_finite(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        anyhow::bail!("{name} must be finite (got {value})");
    }
    if value < 0.0 {
        anyhow::bail!("{name} must be non-negative (got {value})");
    }
    Ok(())
}

/// Handle the `ny diff` command.
///
/// Compares two ONNX models layer-by-layer, identifying divergence points and
/// providing diagnosis of the root cause.
// Justification: Diff command needs both model paths, optional input, tolerance,
// comparison mode, backend, verbosity, and output format — all from CLI arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_diff_command(
    model_a: &Path,
    model_b: &Path,
    input: Option<&Path>,
    tolerance: f32,
    layer_map: Option<&Path>,
    continue_after_divergence: bool,
    diagnose: bool,
    json: bool,
) -> Result<()> {
    use ny_onnx::diff::{diff_models, load_npy, DiffConfig, DiffStatus};

    validate_nonnegative_finite("tolerance", tolerance)?;

    if !json {
        info!("Comparing models layer-by-layer:");
        info!("  Model A: {}", model_a.display());
        info!("  Model B: {}", model_b.display());
        if diagnose {
            info!("  Diagnosis: enabled");
        }
    }

    // Load input if provided
    let input_array = if let Some(input_path) = input {
        if !json {
            info!("  Input: {}", input_path.display());
        }
        Some(load_npy(input_path).context("Failed to load input .npy file")?)
    } else {
        if !json {
            info!("  Input: synthetic zeros");
        }
        None
    };

    let layer_mapping = if let Some(path) = layer_map {
        load_layer_mapping(path)?
    } else {
        HashMap::new()
    };

    let config = DiffConfig {
        tolerance,
        continue_after_divergence,
        input: input_array,
        layer_mapping,
        diagnose,
    };

    let result = diff_models(model_a, model_b, &config)
        .map_err(|e| anyhow::anyhow!("Diff failed: {}", e))?;

    if json {
        // JSON output for programmatic use
        let statuses = result.statuses();
        let layers_json: Vec<_> = result
            .layers
            .iter()
            .zip(statuses.iter())
            .map(|(layer, status)| {
                let status_str = match status {
                    DiffStatus::Ok => "ok",
                    DiffStatus::DriftStarts => "drift_starts",
                    DiffStatus::ExceedsTolerance => "exceeds_tolerance",
                    DiffStatus::ShapeMismatch => "shape_mismatch",
                };
                serde_json::json!({
                    "name": layer.name,
                    "name_b": layer.name_b,
                    "max_diff": layer.max_diff,
                    "mean_diff": layer.mean_diff,
                    "exceeds_tolerance": layer.exceeds_tolerance,
                    "shape_a": layer.shape_a,
                    "shape_b": layer.shape_b,
                    "status": status_str
                })
            })
            .collect();

        // Build diagnosis JSON if available
        let diagnosis_json = result.diagnosis.as_ref().map(|d| {
            serde_json::json!({
                "divergence_layer": d.divergence_layer,
                "layer_type": format!("{:?}", d.layer_type),
                "pattern": format!("{}", d.pattern),
                "explanation": d.explanation,
                "suggestion": d.suggestion,
                "confidence": d.confidence,
                "evidence": d.evidence
            })
        });

        println!(
            "{}",
            serde_json::json!({
                "equivalent": result.is_equivalent(),
                "max_divergence": result.max_divergence,
                "tolerance": result.tolerance,
                "first_bad_layer": result.first_bad_layer,
                "first_bad_layer_name": result.first_bad_layer_name(),
                "drift_start_layer": result.drift_start_layer,
                "suggestion": result.suggestion,
                "diagnosis": diagnosis_json,
                "model_a": model_a.display().to_string(),
                "model_b": model_b.display().to_string(),
                "input": input.map(|p| p.display().to_string()),
                "layers": layers_json
            })
        );
    } else {
        // Human-readable output

        // Display results in table format
        println!("\nLayer-by-Layer Comparison");
        println!("==========================");
        println!("{:<40} | {:<12} | Status", "Layer", "Max Diff");
        println!("{:-<40}-+-{:-<12}-+--------", "", "");

        let statuses = result.statuses();
        for (layer, status) in result.layers.iter().zip(statuses.iter()) {
            let status_str = match status {
                DiffStatus::Ok => "OK",
                DiffStatus::DriftStarts => "DRIFT STARTS HERE",
                DiffStatus::ExceedsTolerance => "EXCEEDS TOLERANCE",
                DiffStatus::ShapeMismatch => "SHAPE MISMATCH",
            };
            println!(
                "{:<40} | {:>12.3e} | {}",
                layer.name, layer.max_diff, status_str
            );
        }

        // Print summary
        println!();
        if result.is_equivalent() {
            println!(
                "EQUIVALENT: Models produce matching outputs within tolerance {:.2e}",
                tolerance
            );
        } else {
            println!(
                "DIVERGENT: Models differ beyond tolerance {:.2e}",
                tolerance
            );

            // Display detailed diagnosis if available
            if let Some(ref diagnosis) = result.diagnosis {
                println!("\nRoot Cause Analysis:");
                println!("--------------------");
                print!("{}", diagnosis.format_report());
            } else {
                // Fall back to simple root cause display
                if let Some(name) = result.first_bad_layer_name() {
                    println!("\nFirst divergence at: {}", name);
                }
                if let Some(suggestion) = &result.suggestion {
                    println!("Suggestion: {}", suggestion);
                }
            }
        }
    }

    Ok(())
}

fn load_layer_mapping(path: &Path) -> Result<HashMap<String, String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read layer mapping file {}", path.display()))?;
    let mapping = match path.extension().and_then(|ext| ext.to_str()) {
        Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse YAML layer map {}", path.display()))?,
        _ => serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON layer map {}", path.display()))?,
    };
    Ok(mapping)
}

fn resolve_center_zeros_onnx_shape(shape: &[i64]) -> Vec<usize> {
    ny_onnx::resolve_dynamic_shape(shape, 1)
}

/// Handle the `ny sensitivity` command.
///
/// Analyzes layer-by-layer sensitivity to input perturbations.
pub(crate) fn handle_sensitivity_command(
    model: &Path,
    epsilon: f32,
    continue_after_overflow: bool,
    threshold: Option<f32>,
    json: bool,
) -> Result<()> {
    use ny_onnx::sensitivity::{analyze_sensitivity, SensitivityConfig};

    validate_nonnegative_finite("epsilon", epsilon)?;
    if let Some(threshold) = threshold {
        validate_nonnegative_finite("threshold", threshold)?;
    }

    info!("Analyzing sensitivity: {}", model.display());

    let config = SensitivityConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    let result = analyze_sensitivity(model, &config)
        .map_err(|e| anyhow::anyhow!("Sensitivity analysis failed: {}", e))?;

    if json {
        // JSON output for programmatic use
        let layers_json: Vec<_> = result
            .layers
            .iter()
            .map(|l| {
                serde_json::json!({
                    "name": l.name,
                    "layer_type": l.layer_type,
                    "input_width": l.input_width,
                    "output_width": l.output_width,
                    "sensitivity": l.sensitivity,
                    "has_overflow": l.has_overflow
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "layers": layers_json,
                "total_sensitivity": result.total_sensitivity,
                "max_sensitivity": result.max_sensitivity,
                "max_sensitivity_layer": result.max_sensitivity_layer
                    .and_then(|i| result.layers.get(i))
                    .map(|l| l.name.as_str()),
                "input_epsilon": result.input_epsilon,
                "final_width": result.final_width,
                "overflow_at_layer": result.overflow_at_layer
                    .and_then(|i| result.layers.get(i))
                    .map(|l| l.name.as_str())
            })
        );
    } else {
        // Human-readable output
        if let Some(thresh) = threshold {
            // Filter to high-sensitivity layers only
            let hot_spots = result.hot_spots(thresh);
            if hot_spots.is_empty() {
                println!("No layers with sensitivity > {:.2} found.", thresh);
            } else {
                println!("High-Sensitivity Layers (sensitivity > {:.2}):", thresh);
                println!("{:-<60}", "");
                for layer in hot_spots {
                    println!("  {:<40} sensitivity={:.2}", layer.name, layer.sensitivity);
                }
            }
        } else {
            // Full summary
            println!("{}", result.summary());
        }
    }

    Ok(())
}

/// Handle the `ny quantize-check` command.
///
/// Analyzes whether ONNX model bounds are safe for float16/int8 quantization.
pub(crate) fn handle_quantize_check_command(
    model: &Path,
    epsilon: f32,
    continue_after_overflow: bool,
    check_float16: bool,
    check_int8: bool,
    json: bool,
) -> Result<()> {
    use ny_onnx::quantize::{analyze_quantization, QuantizeConfig};

    validate_nonnegative_finite("epsilon", epsilon)?;

    info!("Analyzing quantization safety: {}", model.display());

    let config = QuantizeConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    let result = analyze_quantization(model, &config)
        .map_err(|e| anyhow::anyhow!("Quantization analysis failed: {}", e))?;

    if json {
        // JSON output for programmatic use
        let layers_json: Vec<_> = result
            .layers
            .iter()
            .map(|l| {
                let mut obj = serde_json::json!({
                    "name": l.name,
                    "layer_type": l.layer_type,
                    "min_bound": l.min_bound,
                    "max_bound": l.max_bound,
                    "max_abs": l.max_abs,
                    "has_overflow": l.has_overflow
                });
                if check_float16 {
                    obj["float16_safety"] = serde_json::json!(format!("{}", l.float16_safety));
                }
                if check_int8 {
                    obj["int8_safety"] = serde_json::json!(format!("{}", l.int8_safety));
                    obj["int8_scale"] = serde_json::json!(l.int8_scale);
                }
                obj
            })
            .collect();

        let mut output = serde_json::json!({
            "layers": layers_json,
            "input_epsilon": result.input_epsilon
        });
        if check_float16 {
            output["float16_safe"] = serde_json::json!(result.float16_safe);
            output["float16_overflow_count"] = serde_json::json!(result.float16_overflow_count);
            output["denormal_count"] = serde_json::json!(result.denormal_count);
        }
        if check_int8 {
            output["int8_safe"] = serde_json::json!(result.int8_safe);
            output["int8_overflow_count"] = serde_json::json!(result.int8_overflow_count);
        }
        println!("{}", output);
    } else {
        // Human-readable output
        println!("{}", result.summary());

        // Print suggestions
        if check_float16 && !result.float16_safe {
            println!("\nFloat16 Unsafe Layers:");
            for layer in result.float16_unsafe_layers() {
                println!(
                    "  {}: bounds [{:.3e}, {:.3e}]",
                    layer.name, layer.min_bound, layer.max_bound
                );
            }
        }

        if check_int8 && !result.int8_safe {
            println!("\nInt8 Unsafe Layers:");
            for layer in result.int8_unsafe_layers() {
                println!(
                    "  {}: bounds [{:.3e}, {:.3e}]",
                    layer.name, layer.min_bound, layer.max_bound
                );
            }
        }

        if check_float16 && result.denormal_count > 0 {
            println!("\nDenormal Warning Layers:");
            for layer in result.denormal_layers() {
                println!("  {}: values may be in float16 denormal range", layer.name);
            }
        }
    }

    Ok(())
}

/// Handle the `ny profile-bounds` command.
///
/// Profiles bound propagation through a model, identifying layers with
/// excessive bound growth.
// Justification: Profiling command needs model, perturbation, method, backend,
// layer filter, and output format — all specified as CLI arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_profile_bounds_command(
    model: &Path,
    epsilon: f32,
    continue_after_overflow: bool,
    threshold: Option<f32>,
    native: bool,
    json: bool,
    center_zeros: bool,
) -> Result<()> {
    use ny_onnx::profile::{profile_bounds_graph, ProfileConfig};

    validate_nonnegative_finite("epsilon", epsilon)?;
    if let Some(threshold) = threshold {
        validate_nonnegative_finite("threshold", threshold)?;
    }

    info!("Profiling bounds: {}", model.display());

    // Note: input will be set below based on model input shape
    let mut config = ProfileConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    // If center_zeros, we need to create zeros-centered input after getting input shape
    let use_center_zeros = center_zeros;

    // Auto-detect native format based on extension if --native not specified
    let use_native = native || model.is_dir() || {
        let ext = model
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        matches!(
            ext.as_str(),
            "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
        )
    };

    let result = if use_native {
        use ny_onnx::native::NativeModel;

        let native_model = NativeModel::load(model)?;
        let network = &native_model.network;

        info!(
            "Loaded native model: {} ({:?}, {} params)",
            network.name, native_model.config.architecture, network.param_count
        );

        // Convert to GraphNetwork for profiling
        let mut graph_net = native_model.to_graph_network()?;
        // Enable forward-mode LayerNorm for tighter bounds
        let num_modified = graph_net.set_layernorm_forward_mode(true);
        if num_modified > 0 && !json {
            eprintln!(
                "Note: enabled LayerNorm forward-mode for {} LayerNorm nodes. \
                Forward-mode trades strict soundness for tighter bounds. \
                Results may miss worst-case behavior for large perturbations.",
                num_modified
            );
        }

        // Get input shape from network spec
        // Dynamic dims (PyTorch -1, TensorFlow 0) default to 16 for profiling
        let input_shape: Vec<usize> = network
            .inputs
            .first()
            .map(|i| ny_onnx::resolve_dynamic_shape(&i.shape, 16))
            .unwrap_or_else(|| vec![native_model.config.hidden_dim]);

        // If center_zeros, create zeros-centered input for validation
        if use_center_zeros {
            let zeros = ArrayD::zeros(IxDyn(&input_shape));
            config.input = Some(BoundedTensor::from_epsilon(zeros, epsilon)?);
        }

        profile_bounds_graph(&graph_net, &config, &input_shape)
            .map_err(|e| anyhow::anyhow!("Bound profiling failed: {}", e))?
    } else {
        use ny_onnx::{load_onnx, profile::profile_bounds_model};

        // Load ONNX model first to get input shape
        let onnx_model = load_onnx(model)?;

        // If center_zeros, create zeros-centered input for validation
        if use_center_zeros {
            let input_spec = onnx_model
                .network
                .inputs
                .first()
                .ok_or_else(|| anyhow::anyhow!("No input specification in ONNX model"))?;
            let shape = resolve_center_zeros_onnx_shape(&input_spec.shape);
            let zeros = ArrayD::zeros(IxDyn(&shape));
            config.input = Some(BoundedTensor::from_epsilon(zeros, epsilon)?);
        }

        profile_bounds_model(&onnx_model, &config)
            .map_err(|e| anyhow::anyhow!("Bound profiling failed: {}", e))?
    };

    if json {
        // JSON output for programmatic use
        let layers_json: Vec<_> = result
            .layers
            .iter()
            .map(|l| {
                serde_json::json!({
                    "name": l.name,
                    "layer_type": l.layer_type,
                    "input_width": l.input_width,
                    "output_width": l.output_width,
                    "mean_output_width": l.mean_output_width,
                    "median_output_width": l.median_output_width,
                    "growth_ratio": l.growth_ratio,
                    "cumulative_expansion": l.cumulative_expansion,
                    "status": format!("{}", l.status)
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "layers": layers_json,
                "input_epsilon": result.input_epsilon,
                "initial_width": result.initial_width,
                "final_width": result.final_width,
                "total_expansion": result.total_expansion,
                "max_growth_ratio": result.max_growth_ratio,
                "max_growth_layer": result.max_growth_layer
                    .and_then(|i| result.layers.get(i))
                    .map(|l| l.name.as_str()),
                "overflow_at_layer": result.overflow_at_layer
                    .and_then(|i| result.layers.get(i))
                    .map(|l| l.name.as_str()),
                "difficulty_score": result.difficulty_score
            })
        );
    } else {
        // Human-readable output
        if let Some(thresh) = threshold {
            // Filter to high-growth layers only
            let choke_points = result.choke_points(thresh);
            if choke_points.is_empty() {
                println!("No layers with growth > {:.2}x found.", thresh);
            } else {
                println!("Choke Points (growth > {:.2}x):", thresh);
                println!("{:-<60}", "");
                for layer in choke_points {
                    println!(
                        "  {:<40} growth={:.2}x status={}",
                        layer.name, layer.growth_ratio, layer.status
                    );
                }
            }
        } else {
            // Full summary
            println!("{}", result.summary());
        }

        // Print problematic layers
        let problems = result.problematic_layers();
        if !problems.is_empty() {
            println!("\nProblematic Layers (WIDE or worse):");
            for layer in problems {
                println!(
                    "  {}: width={:.3e}, growth={:.2}x, status={}",
                    layer.name, layer.output_width, layer.growth_ratio, layer.status
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{resolve_center_zeros_onnx_shape, validate_nonnegative_finite};

    #[test]
    fn test_resolve_center_zeros_onnx_shape_substitutes_zero_and_negative_dims_2883() {
        assert_eq!(
            resolve_center_zeros_onnx_shape(&[0, 80, -1]),
            vec![1, 80, 1]
        );
    }

    #[test]
    fn analysis_numeric_options_reject_negative_and_non_finite_values() {
        assert!(validate_nonnegative_finite("epsilon", 0.0).is_ok());
        assert!(validate_nonnegative_finite("epsilon", 0.01).is_ok());
        assert!(validate_nonnegative_finite("epsilon", -0.01).is_err());
        assert!(validate_nonnegative_finite("epsilon", f32::NAN).is_err());
        assert!(validate_nonnegative_finite("epsilon", f32::INFINITY).is_err());
    }
}
