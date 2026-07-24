// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-region scoring logic for weak-region mining (#3520).
//!
//! Extracted from `runner.rs` to keep the runner focused on orchestration
//! (loop, sort, winner collection, manifest) and the scoring functions
//! focused on individual region evaluation.

use ny_propagate::types::BoundsProvenance;
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::info;

use crate::analysis_error::AnalysisError;
use crate::profile::{profile_bounds_graph, ProfileConfig};

use super::report::compute_region_id;
use super::types::{WeakRegionHotspot, WeakRegionRecord};

pub(super) struct LinearSweepArgs<'a> {
    pub(super) spec_matrix: &'a ndarray::Array2<f32>,
    pub(super) thresholds: Option<&'a ndarray::Array1<f32>>,
    pub(super) deadline: Option<Instant>,
}

/// Score a region using output-width objective (Packet A/B/D behavior).
pub(super) fn score_region_output_width(
    graph: &ny_propagate::GraphNetwork,
    input: &BoundedTensor,
    region: &super::types::RegionSpec,
    primary_input: &str,
    hotspot_limit: usize,
) -> Result<WeakRegionRecord, AnalysisError> {
    let (output_bounds, provenance) = match graph.propagate_crown_batched_with_provenance(input) {
        Ok(crown_result) => (crown_result.bounds, crown_result.provenance),
        Err(_crown_err) => {
            info!(
                "CROWN backward failed for region '{}', falling back to IBP",
                region.label
            );
            let ibp_bounds = graph
                .propagate_ibp(input)
                .map_err(|e| AnalysisError::propagation("training_signal", e))?;
            (
                ibp_bounds,
                BoundsProvenance::ForwardFallback(
                    ny_propagate::types::CrownIbpFallbackReason::CrownPropagationError,
                ),
            )
        }
    };

    let output_width_max = output_bounds.max_width();
    let output_width_mean = compute_mean_width(&output_bounds);
    let method_actual = match provenance {
        BoundsProvenance::Crown => "batched_crown",
        BoundsProvenance::ForwardFallback(_) => "forward_fallback",
    };
    let hotspots = profile_hotspots(graph, input, hotspot_limit);
    let region_id = compute_region_id(
        primary_input,
        region.lower.shape(),
        region.upper.shape(),
        region.lower.iter().copied(),
        region.upper.iter().copied(),
    );

    Ok(WeakRegionRecord {
        region_id,
        label: region.label.clone(),
        primary_input: primary_input.to_string(),
        lower_shape: region.lower.shape().to_vec(),
        upper_shape: region.upper.shape().to_vec(),
        method_requested: "batched_crown".to_string(),
        method_actual: method_actual.to_string(),
        provenance,
        output_width_max,
        output_width_mean,
        certified_slack_min: None,
        objective_width_max: None,
        objective_width_mean: None,
        top_hotspots: hotspots,
        bounds_file: None,
        metadata: region.metadata.clone(),
    })
}

/// Score a region using spec-guided CROWN with a linear property (Packet C).
pub(super) fn score_region_linear(
    graph: &ny_propagate::GraphNetwork,
    input: &BoundedTensor,
    region: &super::types::RegionSpec,
    primary_input: &str,
    hotspot_limit: usize,
    objective: LinearSweepArgs<'_>,
) -> Result<WeakRegionRecord, AnalysisError> {
    let node_bounds = graph
        .collect_node_bounds(input)
        .map_err(|e| AnalysisError::propagation("training_signal", e))?;

    let crown_result = graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            input,
            objective.spec_matrix,
            None,
            &node_bounds,
            objective.deadline,
        )
        .map_err(|e| AnalysisError::propagation("training_signal", e))?;

    let spec_bounds = &crown_result.bounds;
    let provenance = crown_result.provenance;

    let num_specs = spec_bounds.len();
    let (certified_slack_min, objective_width_max, objective_width_mean) = if num_specs > 0 {
        let mut min_slack = f32::INFINITY;
        let mut max_width = f32::NEG_INFINITY;
        let mut total_width = 0.0_f32;
        for i in 0..num_specs {
            let tau_i = objective.thresholds.map_or(0.0, |t| t[i]);
            let slack = spec_bounds.lower()[[i]] - tau_i;
            min_slack = min_slack.min(slack);
            let w = spec_bounds.upper()[[i]] - spec_bounds.lower()[[i]];
            max_width = max_width.max(w);
            total_width += w;
        }
        (
            Some(min_slack),
            Some(max_width),
            Some(total_width / num_specs as f32),
        )
    } else {
        (None, None, None)
    };

    let graph_output = graph.output_name();
    let (output_width_max, output_width_mean) =
        if let Some(output_bt) = node_bounds.get(graph_output) {
            (output_bt.max_width(), compute_mean_width(output_bt))
        } else {
            (0.0, 0.0)
        };

    let method_actual = match provenance {
        BoundsProvenance::Crown => "spec_guided_crown",
        BoundsProvenance::ForwardFallback(_) => "forward_fallback",
    };
    let hotspots = profile_hotspots(graph, input, hotspot_limit);
    let region_id = compute_region_id(
        primary_input,
        region.lower.shape(),
        region.upper.shape(),
        region.lower.iter().copied(),
        region.upper.iter().copied(),
    );

    Ok(WeakRegionRecord {
        region_id,
        label: region.label.clone(),
        primary_input: primary_input.to_string(),
        lower_shape: region.lower.shape().to_vec(),
        upper_shape: region.upper.shape().to_vec(),
        method_requested: "spec_guided_crown".to_string(),
        method_actual: method_actual.to_string(),
        provenance,
        output_width_max,
        output_width_mean,
        certified_slack_min,
        objective_width_max,
        objective_width_mean,
        top_hotspots: hotspots,
        bounds_file: None,
        metadata: region.metadata.clone(),
    })
}

fn compute_mean_width(bt: &BoundedTensor) -> f32 {
    let widths = bt.width();
    let len = widths.len();
    if len > 0 {
        widths.iter().sum::<f32>() / len as f32
    } else {
        0.0
    }
}

fn profile_hotspots(
    graph: &ny_propagate::GraphNetwork,
    input: &BoundedTensor,
    hotspot_limit: usize,
) -> Vec<WeakRegionHotspot> {
    let epsilon = (0.5 * input.max_width()).max(1e-6);
    let profile_config = ProfileConfig {
        epsilon,
        continue_after_overflow: true,
        input: Some(input.clone()),
    };
    match profile_bounds_graph(graph, &profile_config, input.shape()) {
        Ok(profile) => extract_hotspots(&profile, hotspot_limit),
        Err(_) => Vec::new(),
    }
}

/// Extract hotspots from a profile result.
///
/// Prefers `problematic_layers()` (Wide, VeryWide, Overflow); if none,
/// falls back to `layers_by_growth().take(limit)`.
fn extract_hotspots(
    profile: &crate::profile::ProfileResult,
    limit: usize,
) -> Vec<WeakRegionHotspot> {
    let problematic = profile.problematic_layers();
    let source: Vec<&crate::profile::LayerProfile> = if problematic.is_empty() {
        profile.layers_by_growth().into_iter().take(limit).collect()
    } else {
        problematic.into_iter().take(limit).collect()
    };

    source
        .into_iter()
        .map(|lp| WeakRegionHotspot {
            name: lp.name.clone(),
            layer_type: lp.layer_type.clone(),
            max_width: lp.output_width,
            mean_width: lp.mean_output_width,
            growth_ratio: lp.growth_ratio,
            status: lp.status.to_string(),
        })
        .collect()
}
