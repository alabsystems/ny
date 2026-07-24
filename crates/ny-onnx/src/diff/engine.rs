// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::compare::compare_arrays;
use super::diagnosis::{diagnose_divergence, suggest_root_cause, DiagnosisContext};
use super::inference::{
    run_inference, run_inference_bytes, run_inference_with_intermediates,
    run_inference_with_intermediates_bytes,
};
use super::io::{load_model_info, load_model_info_bytes};
use super::matching::normalize_layer_name;
use super::{DiffConfig, DiffError, DiffResult};
use crate::LayerSpec;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Perform a full diff between two ONNX models.
///
/// This is the main entry point for comparing models. It extracts all intermediate
/// tensors from both models and compares them layer by layer.
pub fn diff_models(
    path_a: impl AsRef<Path>,
    path_b: impl AsRef<Path>,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    info!("Loading model A: {}", path_a.as_ref().display());
    let info_a = load_model_info(&path_a)?;

    info!("Loading model B: {}", path_b.as_ref().display());
    let info_b = load_model_info(&path_b)?;

    // Check input compatibility
    if info_a.inputs.is_empty() || info_b.inputs.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let input_a = &info_a.inputs[0];
    let _input_b = &info_b.inputs[0]; // Could validate shape match in future

    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        // Use zeros with model A's input shape
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    debug!("Input shape: {:?}", input.shape());

    // Run inference with intermediate outputs on both models
    info!("Running inference on model A (extracting all intermediate outputs)");
    let outputs_a = run_inference_with_intermediates(&path_a, &input)?;

    info!("Running inference on model B (extracting all intermediate outputs)");
    let outputs_b = run_inference_with_intermediates(&path_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    info!(
        "Model A: {} tensors, Model B: {} tensors",
        outputs_a.len(),
        outputs_b.len()
    );

    // Build tensor name to layer spec mapping for suggestions
    let mut tensor_to_layer_a: HashMap<String, &LayerSpec> = HashMap::new();
    for layer in &info_a.layers {
        for output in &layer.outputs {
            tensor_to_layer_a.insert(output.clone(), layer);
        }
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;
    let mut first_bad_layer_spec: Option<&LayerSpec> = None;

    // Strategy: iterate through model A's layer outputs in order
    // and match them with model B's outputs
    for layer_a in &info_a.layers {
        for output_name_a in &layer_a.outputs {
            if output_name_a.is_empty() {
                continue;
            }

            // Get output from model A
            let out_a = match outputs_a.get(output_name_a) {
                Some(arr) => arr,
                None => continue, // Skip tensors we couldn't extract
            };

            // Try to find matching output in model B
            // First check explicit mapping
            let output_name_b = config
                .layer_mapping
                .get(output_name_a)
                .cloned()
                .or_else(|| {
                    // Try exact match
                    if outputs_b.contains_key(output_name_a) {
                        Some(output_name_a.clone())
                    } else {
                        // Try normalized name matching
                        let normalized_a = normalize_layer_name(output_name_a);
                        outputs_b
                            .keys()
                            .find(|name_b| normalize_layer_name(name_b) == normalized_a)
                            .cloned()
                    }
                });

            let (out_b, matched_name_b) = match output_name_b {
                Some(ref name_b) => match outputs_b.get(name_b) {
                    Some(arr) => (arr, Some(name_b.clone())),
                    None => continue, // Matched name but no output
                },
                None => continue, // No match found
            };

            // Compare arrays
            let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
            comparison.name = output_name_a.clone();
            comparison.name_b = if matched_name_b.as_ref() == Some(output_name_a) {
                None
            } else {
                matched_name_b
            };

            max_divergence = max_divergence.max(comparison.max_diff);

            let idx = layers.len();

            // Detect first bad layer
            if comparison.exceeds_tolerance && first_bad_layer.is_none() {
                first_bad_layer = Some(idx);
                first_bad_layer_spec = tensor_to_layer_a.get(output_name_a).copied();
            }

            // Detect drift start (within 10x tolerance but above tolerance/10)
            if comparison.max_diff > config.tolerance / 10.0
                && comparison.max_diff <= config.tolerance * 10.0
                && drift_start_layer.is_none()
            {
                drift_start_layer = Some(idx);
            }

            layers.push(comparison);

            // Stop early if not continuing after divergence
            if !config.continue_after_divergence && first_bad_layer.is_some() {
                break;
            }
        }

        if !config.continue_after_divergence && first_bad_layer.is_some() {
            break;
        }
    }

    // If we didn't match any layers, fall back to comparing final outputs only
    if layers.is_empty() {
        warn!("No intermediate layers matched between models, comparing final outputs only");
        return diff_models_final_only(path_a, path_b, config);
    }

    // Generate suggestion based on the layer where divergence starts
    let suggestion = first_bad_layer_spec.and_then(suggest_root_cause);

    // Generate detailed diagnosis if enabled and there's a divergence
    let diagnosis = if config.diagnose {
        first_bad_layer.map(|bad_layer| {
            let ctx = DiagnosisContext {
                outputs_a: &outputs_a,
                outputs_b: &outputs_b,
                layers_a: &info_a.layers,
                comparisons: &layers,
                tolerance: config.tolerance,
            };
            diagnose_divergence(&ctx, bad_layer, first_bad_layer_spec)
        })
    } else {
        None
    };

    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis,
    })
}

/// Perform a full diff between two in-memory ONNX models.
pub fn diff_models_bytes(
    name_a: &str,
    model_a: &[u8],
    name_b: &str,
    model_b: &[u8],
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    info!("Loading model A (memory): {}", name_a);
    let info_a = load_model_info_bytes(name_a, model_a)?;

    info!("Loading model B (memory): {}", name_b);
    let info_b = load_model_info_bytes(name_b, model_b)?;

    if info_a.inputs.is_empty() || info_b.inputs.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let input_a = &info_a.inputs[0];
    let _input_b = &info_b.inputs[0];

    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    debug!("Input shape: {:?}", input.shape());

    info!("Running inference on model A (memory, intermediates)");
    let outputs_a = run_inference_with_intermediates_bytes(model_a, &input)?;

    info!("Running inference on model B (memory, intermediates)");
    let outputs_b = run_inference_with_intermediates_bytes(model_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    info!(
        "Model A: {} tensors, Model B: {} tensors",
        outputs_a.len(),
        outputs_b.len()
    );

    let mut tensor_to_layer_a: HashMap<String, &LayerSpec> = HashMap::new();
    for layer in &info_a.layers {
        for output in &layer.outputs {
            tensor_to_layer_a.insert(output.clone(), layer);
        }
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;
    let mut first_bad_layer_spec: Option<&LayerSpec> = None;

    for layer_a in &info_a.layers {
        for output_name_a in &layer_a.outputs {
            if output_name_a.is_empty() {
                continue;
            }

            let out_a = match outputs_a.get(output_name_a) {
                Some(arr) => arr,
                None => continue,
            };

            let output_name_b = config
                .layer_mapping
                .get(output_name_a)
                .cloned()
                .or_else(|| {
                    if outputs_b.contains_key(output_name_a) {
                        Some(output_name_a.clone())
                    } else {
                        let normalized_a = normalize_layer_name(output_name_a);
                        outputs_b
                            .keys()
                            .find(|name_b| normalize_layer_name(name_b) == normalized_a)
                            .cloned()
                    }
                });

            let (out_b, matched_name_b) = match output_name_b {
                Some(ref name_b) => match outputs_b.get(name_b) {
                    Some(arr) => (arr, Some(name_b.clone())),
                    None => continue,
                },
                None => continue,
            };

            let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
            comparison.name = output_name_a.clone();
            comparison.name_b = if matched_name_b.as_ref() == Some(output_name_a) {
                None
            } else {
                matched_name_b
            };

            max_divergence = max_divergence.max(comparison.max_diff);

            let idx = layers.len();

            if comparison.exceeds_tolerance && first_bad_layer.is_none() {
                first_bad_layer = Some(idx);
                first_bad_layer_spec = tensor_to_layer_a.get(output_name_a).copied();
            }

            if comparison.max_diff > config.tolerance / 10.0
                && comparison.max_diff <= config.tolerance * 10.0
                && drift_start_layer.is_none()
            {
                drift_start_layer = Some(idx);
            }

            layers.push(comparison);

            if !config.continue_after_divergence && first_bad_layer.is_some() {
                break;
            }
        }

        if !config.continue_after_divergence && first_bad_layer.is_some() {
            break;
        }
    }

    if layers.is_empty() {
        warn!("No intermediate layers matched between models, comparing final outputs only");
        return diff_models_bytes_final_only(model_a, model_b, &info_a, config);
    }

    let suggestion = first_bad_layer_spec.and_then(suggest_root_cause);

    let diagnosis = if config.diagnose {
        first_bad_layer.map(|bad_layer| {
            let ctx = DiagnosisContext {
                outputs_a: &outputs_a,
                outputs_b: &outputs_b,
                layers_a: &info_a.layers,
                comparisons: &layers,
                tolerance: config.tolerance,
            };
            diagnose_divergence(&ctx, bad_layer, first_bad_layer_spec)
        })
    } else {
        None
    };

    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis,
    })
}

/// Fallback: compare only final outputs (used when intermediate matching fails).
fn diff_models_final_only(
    path_a: impl AsRef<Path>,
    path_b: impl AsRef<Path>,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    let info_a = load_model_info(&path_a)?;
    let _info_b = load_model_info(&path_b)?;

    let input_a = &info_a.inputs[0];

    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    let outputs_a = run_inference(&path_a, &input)?;
    let outputs_b = run_inference(&path_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;

    for (i, (out_a, out_b)) in outputs_a.iter().zip(outputs_b.iter()).enumerate() {
        let output_name = info_a
            .outputs
            .get(i)
            .map(|o| o.name.clone())
            .unwrap_or_else(|| format!("output_{}", i));

        let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
        comparison.name = output_name;

        max_divergence = max_divergence.max(comparison.max_diff);

        if comparison.exceeds_tolerance && first_bad_layer.is_none() {
            first_bad_layer = Some(i);
        }

        if comparison.max_diff > config.tolerance / 10.0
            && comparison.max_diff <= config.tolerance * 10.0
            && drift_start_layer.is_none()
        {
            drift_start_layer = Some(i);
        }

        layers.push(comparison);
    }

    let suggestion = first_bad_layer
        .and_then(|_| info_a.layers.last())
        .and_then(suggest_root_cause);

    // Note: Diagnosis is not available in final-only mode since we lack intermediate data
    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis: None,
    })
}

fn diff_models_bytes_final_only(
    model_a: &[u8],
    model_b: &[u8],
    info_a: &super::ModelInfo,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    let input_a = &info_a.inputs[0];

    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let shape: Vec<usize> = input_a
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        ArrayD::zeros(IxDyn(&shape))
    };

    let outputs_a = run_inference_bytes(model_a, &input)?;
    let outputs_b = run_inference_bytes(model_b, &input)?;

    if outputs_a.is_empty() || outputs_b.is_empty() {
        return Err(DiffError::NoLayers);
    }

    let mut layers = Vec::new();
    let mut first_bad_layer = None;
    let mut drift_start_layer = None;
    let mut max_divergence: f32 = 0.0;

    for (i, (out_a, out_b)) in outputs_a.iter().zip(outputs_b.iter()).enumerate() {
        let output_name = info_a
            .outputs
            .get(i)
            .map(|o| o.name.clone())
            .unwrap_or_else(|| format!("output_{}", i));

        let mut comparison = compare_arrays(out_a, out_b, config.tolerance);
        comparison.name = output_name;

        max_divergence = max_divergence.max(comparison.max_diff);

        if comparison.exceeds_tolerance && first_bad_layer.is_none() {
            first_bad_layer = Some(i);
        }

        if comparison.max_diff > config.tolerance / 10.0
            && comparison.max_diff <= config.tolerance * 10.0
            && drift_start_layer.is_none()
        {
            drift_start_layer = Some(i);
        }

        layers.push(comparison);
    }

    let suggestion = first_bad_layer
        .and_then(|_| info_a.layers.last())
        .and_then(suggest_root_cause);

    Ok(DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer,
        max_divergence,
        tolerance: config.tolerance,
        suggestion,
        diagnosis: None,
    })
}
