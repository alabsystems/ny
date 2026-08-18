// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX model bound profiling.

use crate::{load_onnx, OnnxModel};
use ny_propagate::BoundPropagation;
use std::path::Path;
use tracing::{debug, info};

use super::stats::{difficulty_score, make_unit_variance_input, median};
use super::types::{BoundStatus, LayerProfile, ProfileConfig, ProfileError, ProfileResult};
use crate::analysis_error::validate_analysis_epsilon;

/// Analyze a model's bound profile using the normalized `analyze_*` verb
/// family.
pub fn analyze_profile(
    path: impl AsRef<Path>,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    profile_bounds(path, config)
}

/// Profile bounds of a model loaded from ONNX file.
pub fn profile_bounds(
    path: impl AsRef<Path>,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model = load_onnx(path.as_ref()).map_err(|e| ProfileError::load("profile", e))?;

    profile_bounds_model(&onnx_model, config)
}

/// Analyze an already-loaded ONNX model's bound profile using the normalized
/// `analyze_*` verb family.
pub fn analyze_profile_model(
    model: &OnnxModel,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    profile_bounds_model(model, config)
}

/// Profile bounds of an already-loaded ONNX model.
pub fn profile_bounds_model(
    model: &OnnxModel,
    config: &ProfileConfig,
) -> Result<ProfileResult, ProfileError> {
    validate_analysis_epsilon("profile", config.epsilon)?;

    // Convert to propagate network
    let network = model
        .to_propagate_network()
        .map_err(|e| ProfileError::propagation("profile", e))?;

    if network.layers().is_empty() {
        return Err(ProfileError::no_layers("profile"));
    }

    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let input_spec = model.network.inputs.first().ok_or_else(|| {
            ProfileError::invalid_input_shape("profile", "No input specification")
        })?;

        let shape: Vec<usize> = input_spec
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();

        // Use unit-variance input to avoid artificial amplification in LayerNorm/RMSNorm
        make_unit_variance_input(&shape, config.epsilon)
            .map_err(|e| ProfileError::propagation("profile", e))?
    };

    let initial_width = input.max_width();

    info!(
        "Starting bound profile with input shape {:?}, epsilon {}, initial width {}",
        input.shape(),
        config.epsilon,
        initial_width
    );

    // Track layer-by-layer bounds
    let mut layers = Vec::new();
    let mut current = input;
    let mut max_growth_ratio: f32 = 1.0;
    let mut max_growth_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;
    // Once continuation substitutes a layer's input after a propagation error,
    // every later sequential result is based on a graph state that never
    // existed. Keep that diagnostic taint sticky instead of reporting a
    // downstream layer as tight/stable.
    let mut propagation_failed = false;

    for (i, (layer, spec)) in network
        .layers()
        .iter()
        .zip(model.network.layers.iter())
        .enumerate()
    {
        let input_width = current.max_width();

        // Propagate through this layer
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(ProfileError::propagation("profile", e));
                }
                if overflow_at_layer.is_none() {
                    overflow_at_layer = Some(i);
                }
                propagation_failed = true;
                current.clone()
            }
        };

        let output_width = output.max_width();
        let widths: Vec<f32> = output.width().iter().cloned().collect();
        let mean_width = widths.iter().sum::<f32>() / widths.len().max(1) as f32;
        let median_width = median(&widths);

        // Calculate growth ratio
        let growth_ratio = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else {
            1.0
        };

        // Track max growth
        if growth_ratio > max_growth_ratio && growth_ratio.is_finite() {
            max_growth_ratio = growth_ratio;
            max_growth_layer = Some(i);
        }

        // Calculate cumulative expansion from input
        let cumulative_expansion = if initial_width > 0.0 && initial_width.is_finite() {
            output_width / initial_width
        } else {
            1.0
        };

        // Determine status
        let has_overflow = propagation_failed || !output_width.is_finite();
        let status = if has_overflow {
            if overflow_at_layer.is_none() {
                overflow_at_layer = Some(i);
            }
            BoundStatus::Overflow
        } else {
            BoundStatus::from_width(output_width, config.epsilon)
        };

        layers.push(LayerProfile {
            name: spec.name.clone(),
            layer_type: format!("{:?}", spec.layer_type),
            input_width,
            output_width,
            mean_output_width: mean_width,
            median_output_width: median_width,
            growth_ratio,
            cumulative_expansion,
            output_shape: output.shape().to_vec(),
            num_elements: output.lower().len(),
            status,
        });

        debug!(
            "Layer {}: width {} -> {}, growth {:.2}x",
            spec.name, input_width, output_width, growth_ratio
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
    }

    let final_width = current.max_width();
    let total_expansion = if initial_width > 0.0 && initial_width.is_finite() {
        final_width / initial_width
    } else {
        1.0
    };

    let difficulty = difficulty_score(
        total_expansion,
        max_growth_ratio,
        overflow_at_layer.is_some(),
    );

    Ok(ProfileResult {
        layers,
        input_epsilon: config.epsilon,
        initial_width,
        final_width,
        total_expansion,
        max_growth_layer,
        max_growth_ratio,
        overflow_at_layer,
        difficulty_score: difficulty,
    })
}
