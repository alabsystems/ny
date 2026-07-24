// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::EcapaCompositionBoundary;
use super::super::subgraph::extract_single_input_subgraph;
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StageProvenance {
    pub(crate) node_count: usize,
    pub(crate) crown_ibp_tightened_count: usize,
    pub(crate) ibp_fallback_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EcapaStageResult {
    pub(crate) x2_bounds: BoundedTensor,
    pub(crate) x3_bounds: BoundedTensor,
    pub(crate) x4_bounds: BoundedTensor,
    pub(crate) mfa_bounds: BoundedTensor,
    pub(crate) stage_provenances: [StageProvenance; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct EcapaCosineResult {
    pub(crate) dot_lower: f32,
    pub(crate) dot_upper: f32,
    pub(crate) normsq_lower: f32,
    pub(crate) normsq_upper: f32,
    pub(crate) distance_upper: f32,
    pub(crate) nonvacuous: bool,
    pub(crate) stage_result: EcapaStageResult,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarCosineBounds {
    pub(crate) dot_lower: f32,
    pub(crate) dot_upper: f32,
    pub(crate) normsq_lower: f32,
    pub(crate) normsq_upper: f32,
    pub(crate) distance_upper: f32,
    pub(crate) nonvacuous: bool,
}

pub(crate) fn log_stage_fallback_events(label: &str, result: &GraphCrownIbpBoundsResult) {
    for event in result.fallback_events.iter().take(5) {
        eprintln!(
            "{label}: fallback_event[{}] layer_type={} reason={:?} details={}",
            event.layer_index, event.layer_type, event.reason, event.details
        );
    }
}

pub(crate) fn ensure_bounded_tensor_finite_and_ordered(
    bounds: &BoundedTensor,
    label: &str,
) -> Result<(), String> {
    for (idx, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(format!(
                "{label}: non-finite bounds at dim {idx}: lower={lower}, upper={upper}"
            ));
        }
        if lower > upper {
            return Err(format!(
                "{label}: inverted bounds at dim {idx}: lower={lower}, upper={upper}"
            ));
        }
    }
    Ok(())
}

fn ensure_scalar_bounds_finite_and_ordered(
    lower: f32,
    upper: f32,
    label: &str,
) -> Result<(), String> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(format!(
            "{label}: non-finite scalar bounds [{lower}, {upper}]"
        ));
    }
    if lower > upper {
        return Err(format!(
            "{label}: inverted scalar bounds [{lower}, {upper}]"
        ));
    }
    Ok(())
}

pub(crate) fn width_reduction_pct(before: f32, after: f32) -> f32 {
    if before > 0.0 {
        (1.0 - after / before) * 100.0
    } else {
        0.0
    }
}

pub(crate) fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(upper, lower)| upper - lower)
        .sum()
}

pub(crate) fn stage_provenance(result: &GraphCrownIbpBoundsResult) -> StageProvenance {
    StageProvenance {
        node_count: result.bounds.len(),
        crown_ibp_tightened_count: result
            .provenance
            .values()
            .filter(|provenance| matches!(provenance, BoundsProvenance::Crown))
            .count(),
        ibp_fallback_count: result
            .provenance
            .values()
            .filter(|provenance| provenance.is_fallback())
            .count(),
    }
}

pub(crate) fn output_bounds_from_map(
    bounds: &HashMap<String, BoundedTensor>,
    output_name: &str,
    label: &str,
) -> Result<BoundedTensor, String> {
    bounds
        .get(output_name)
        .cloned()
        .ok_or_else(|| format!("{label}: missing output bounds for '{output_name}'"))
}

pub(crate) fn output_bounds_from_crown_result(
    result: &GraphCrownIbpBoundsResult,
    output_name: &str,
    label: &str,
) -> Result<BoundedTensor, String> {
    output_bounds_from_map(&result.bounds, output_name, label)
}

pub(crate) fn extract_ecapa_stage_graphs(
    graph: &GraphNetwork,
    boundary: &EcapaCompositionBoundary,
) -> Result<[GraphNetwork; 3], String> {
    let stage_a = extract_single_input_subgraph(
        graph,
        ny_propagate::NETWORK_INPUT,
        &boundary.block_outputs[0],
    )?;
    let stage_b = extract_single_input_subgraph(
        graph,
        &boundary.block_outputs[0],
        &boundary.block_outputs[1],
    )?;
    let stage_c = extract_single_input_subgraph(
        graph,
        &boundary.block_outputs[1],
        &boundary.block_outputs[2],
    )?;
    Ok([stage_a, stage_b, stage_c])
}

pub(crate) fn scalar_spec_bounds_with_node_bounds_and_deadline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    label: &str,
    deadline_secs: u64,
) -> Result<(f32, f32), String> {
    let spec = ndarray::arr2(&[[1.0_f32]]);
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let crown = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
            input,
            &spec,
            None,
            node_bounds,
            Some(deadline),
        )
        .map_err(|e| format!("{label}: spec-guided CROWN failed: {e}"))?;
    let flat = crown.flatten();
    if flat.lower().len() != 1 {
        return Err(format!(
            "{label}: expected scalar output, got shape {:?}",
            flat.lower().shape()
        ));
    }
    let lower = flat.lower()[0];
    let upper = flat.upper()[0];
    ensure_scalar_bounds_finite_and_ordered(lower, upper, label)?;
    Ok((lower, upper))
}

pub(crate) fn run_monolithic_ibp_cosine_bounds(
    input: &BoundedTensor,
    spec_deadline_secs: u64,
) -> Result<ScalarCosineBounds, String> {
    let base_graph = avoice_speaker_encoder_graph();
    let encoder_output_name = base_graph.output_name().to_string();
    let encoder_bounds = base_graph
        .collect_node_bounds(input)
        .map_err(|e| format!("monolithic encoder IBP failed: {e}"))?;
    let (dot_graph, norm_sq_graph, _) = build_speaker_cosine_component_graphs();
    let dot_node_bounds =
        build_component_node_bounds(&dot_graph, &encoder_bounds, &encoder_output_name);
    let normsq_node_bounds =
        build_component_node_bounds(&norm_sq_graph, &encoder_bounds, &encoder_output_name);
    let (dot_lower, dot_upper) = scalar_spec_bounds_with_node_bounds_and_deadline(
        &dot_graph,
        input,
        &dot_node_bounds,
        "monolithic dot",
        spec_deadline_secs,
    )?;
    let (normsq_lower, normsq_upper) = scalar_spec_bounds_with_node_bounds_and_deadline(
        &norm_sq_graph,
        input,
        &normsq_node_bounds,
        "monolithic normsq",
        spec_deadline_secs,
    )?;
    let (distance_upper, nonvacuous) = speaker_cosine_distance_upper(dot_lower, normsq_upper);
    Ok(ScalarCosineBounds {
        dot_lower,
        dot_upper,
        normsq_lower,
        normsq_upper,
        distance_upper,
        nonvacuous,
    })
}
