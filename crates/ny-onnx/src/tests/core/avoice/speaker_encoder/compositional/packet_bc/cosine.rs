// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::discover_ecapa_composition_boundary;
use super::super::subgraph::extract_single_input_subgraph;
use super::alpha::run_ecapa_stage_local_alpha_crown;
use super::core::{
    scalar_spec_bounds_with_node_bounds_and_deadline, EcapaCosineResult, EcapaStageResult,
};
use super::stage_local::run_ecapa_stage_local_crown_ibp;
use super::*;

/// Budget (seconds) for CROWN-IBP tightening on each suffix subgraph.
///
/// The suffix graph (MFA concat -> cosine scalar output) has ~20-30 nodes
/// including attentive statistics pooling. Tightening intermediates here
/// before the spec-guided CROWN pass gives tighter linearization at each
/// suffix node, especially Conv1d and Dense layers that CROWN can handle.
/// Softmax/Sqrt nodes may still fall back to IBP, but tighter surrounding
/// bounds improve the overall quality.
const SUFFIX_CROWN_IBP_BUDGET_SECS: u64 = 10;

pub(crate) fn cosine_bounds_from_stage_result(
    stage_result: &EcapaStageResult,
    spec_deadline_secs: u64,
    label_prefix: &str,
) -> Result<(f32, f32, f32, f32), String> {
    let (dot_graph, norm_sq_graph, _) = build_speaker_cosine_component_graphs();
    let dot_boundary = discover_ecapa_composition_boundary(&dot_graph)?;
    let normsq_boundary = discover_ecapa_composition_boundary(&norm_sq_graph)?;
    if dot_boundary.mfa_concat != normsq_boundary.mfa_concat {
        return Err(format!(
            "{label_prefix}: dot and normsq MFA concat mismatch: '{}' vs '{}'",
            dot_boundary.mfa_concat, normsq_boundary.mfa_concat
        ));
    }
    let dot_suffix = extract_single_input_subgraph(
        &dot_graph,
        &dot_boundary.mfa_concat,
        dot_graph.output_name(),
    )?;
    let normsq_suffix = extract_single_input_subgraph(
        &norm_sq_graph,
        &normsq_boundary.mfa_concat,
        norm_sq_graph.output_name(),
    )?;

    // Step 1: IBP forward on suffix graphs.
    let dot_ibp_bounds = dot_suffix
        .collect_node_bounds(&stage_result.mfa_bounds)
        .map_err(|e| format!("{label_prefix} dot suffix IBP failed: {e}"))?;
    let normsq_ibp_bounds = normsq_suffix
        .collect_node_bounds(&stage_result.mfa_bounds)
        .map_err(|e| format!("{label_prefix} normsq suffix IBP failed: {e}"))?;

    // Step 2: CROWN-IBP tighten suffix intermediates (#3499).
    //
    // The suffix includes attentive statistics pooling (Softmax, Sqrt) and
    // linear layers (Conv1d, Dense). CROWN-IBP tightening on this small
    // subgraph (~20-30 nodes) produces tighter intermediates for the
    // spec-guided CROWN backward pass, improving the scalar dot/normsq bounds.
    let dot_tighten_deadline = Instant::now() + Duration::from_secs(SUFFIX_CROWN_IBP_BUDGET_SECS);
    let dot_node_bounds = dot_suffix
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &stage_result.mfa_bounds,
            dot_ibp_bounds,
            Some(dot_tighten_deadline),
        )
        .map_err(|e| format!("{label_prefix} dot suffix CROWN-IBP tightening failed: {e}"))?
        .bounds;
    let normsq_tighten_deadline =
        Instant::now() + Duration::from_secs(SUFFIX_CROWN_IBP_BUDGET_SECS);
    let normsq_node_bounds = normsq_suffix
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &stage_result.mfa_bounds,
            normsq_ibp_bounds,
            Some(normsq_tighten_deadline),
        )
        .map_err(|e| format!("{label_prefix} normsq suffix CROWN-IBP tightening failed: {e}"))?
        .bounds;

    // Step 3: Spec-guided CROWN with tightened intermediates.
    let (dot_lower, dot_upper) = scalar_spec_bounds_with_node_bounds_and_deadline(
        &dot_suffix,
        &stage_result.mfa_bounds,
        &dot_node_bounds,
        &format!("{label_prefix} dot"),
        spec_deadline_secs,
    )?;
    let (normsq_lower, normsq_upper) = scalar_spec_bounds_with_node_bounds_and_deadline(
        &normsq_suffix,
        &stage_result.mfa_bounds,
        &normsq_node_bounds,
        &format!("{label_prefix} normsq"),
        spec_deadline_secs,
    )?;
    Ok((dot_lower, dot_upper, normsq_lower, normsq_upper))
}

pub(crate) fn run_ecapa_alpha_compositional_cosine_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_config: &AlphaCrownConfig,
    spec_deadline_secs: u64,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> Result<EcapaCosineResult, String> {
    let stage_result = run_ecapa_stage_local_alpha_crown(graph, input, alpha_config, engine)?;
    let (dot_lower, dot_upper, normsq_lower, normsq_upper) =
        cosine_bounds_from_stage_result(&stage_result, spec_deadline_secs, "alpha")?;
    let (distance_upper, nonvacuous) = speaker_cosine_distance_upper(dot_lower, normsq_upper);
    Ok(EcapaCosineResult {
        dot_lower,
        dot_upper,
        normsq_lower,
        normsq_upper,
        distance_upper,
        nonvacuous,
        stage_result,
    })
}

pub(crate) fn run_ecapa_compositional_cosine_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    per_stage_deadline_secs: u64,
    spec_deadline_secs: u64,
) -> Result<EcapaCosineResult, String> {
    let stage_result = run_ecapa_stage_local_crown_ibp(graph, input, per_stage_deadline_secs)?;
    let (dot_lower, dot_upper, normsq_lower, normsq_upper) =
        cosine_bounds_from_stage_result(&stage_result, spec_deadline_secs, "compositional")?;
    let (distance_upper, nonvacuous) = speaker_cosine_distance_upper(dot_lower, normsq_upper);

    Ok(EcapaCosineResult {
        dot_lower,
        dot_upper,
        normsq_lower,
        normsq_upper,
        distance_upper,
        nonvacuous,
        stage_result,
    })
}
