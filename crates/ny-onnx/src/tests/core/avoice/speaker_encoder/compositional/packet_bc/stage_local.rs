// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::discover_ecapa_composition_boundary;
use super::super::subgraph::concat_mfa_block_bounds;
use super::core::{
    ensure_bounded_tensor_finite_and_ordered, extract_ecapa_stage_graphs,
    log_stage_fallback_events, output_bounds_from_crown_result, output_bounds_from_map,
    stage_provenance, total_bound_width, width_reduction_pct, EcapaStageResult, StageProvenance,
};
use super::*;

pub(crate) fn collect_ecapa_stage_local_crown_ibp(
    stage: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
    per_stage_deadline_secs: u64,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> Result<GraphCrownIbpBoundsResult, String> {
    let output_name = stage.output_name().to_string();
    let ibp_bounds = stage
        .collect_node_bounds(input)
        .map_err(|e| format!("{label}: IBP node-bound collection failed: {e}"))?;
    let stage_total_ibp_width: f32 = ibp_bounds.values().map(total_bound_width).sum();
    let ibp_output = output_bounds_from_map(&ibp_bounds, &output_name, label)?;
    let deadline = Instant::now() + Duration::from_secs(per_stage_deadline_secs);
    let crown_result = stage
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine(
            input,
            ibp_bounds,
            Some(deadline),
            engine,
        )
        .map_err(|e| format!("{label}: CROWN-IBP collection failed: {e}"))?;
    let stage_total_tightened_width: f32 =
        crown_result.bounds.values().map(total_bound_width).sum();
    let output_bounds = output_bounds_from_crown_result(&crown_result, &output_name, label)?;
    ensure_bounded_tensor_finite_and_ordered(&output_bounds, &format!("{label} output"))?;

    let ibp_width = ibp_output.max_width();
    let tightened_width = output_bounds.max_width();
    let reduction_pct = width_reduction_pct(ibp_width, tightened_width);
    let total_reduction_pct =
        width_reduction_pct(stage_total_ibp_width, stage_total_tightened_width);
    let provenance = stage_provenance(&crown_result);
    let output_provenance = crown_result
        .provenance_for_node(&output_name)
        .expect("CROWN-IBP provenance map should contain the output node");
    eprintln!(
        "{label}: nodes={}, tightened={}, fallback={}, output_provenance={output_provenance:?}, max_width {:.6} -> {:.6} ({reduction_pct:.1}% reduction), total_width {:.6} -> {:.6} ({total_reduction_pct:.1}% reduction)",
        provenance.node_count,
        provenance.crown_ibp_tightened_count,
        provenance.ibp_fallback_count,
        ibp_width,
        tightened_width,
        stage_total_ibp_width,
        stage_total_tightened_width,
    );
    if output_provenance.is_fallback() {
        log_stage_fallback_events(label, &crown_result);
    }

    Ok(crown_result)
}

fn run_ecapa_stage_local_crown_ibp_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    per_stage_deadline_secs: u64,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> Result<EcapaStageResult, String> {
    let boundary = discover_ecapa_composition_boundary(graph)?;
    let [stage_a, stage_b, stage_c] = extract_ecapa_stage_graphs(graph, &boundary)?;
    let mut stage_input = input.clone();
    let mut tightened_outputs = Vec::with_capacity(3);
    let mut provenances = Vec::with_capacity(3);

    for (label, stage) in [
        ("stage_a", &stage_a),
        ("stage_b", &stage_b),
        ("stage_c", &stage_c),
    ] {
        let output_name = stage.output_name().to_string();
        let crown_result = collect_ecapa_stage_local_crown_ibp(
            stage,
            &stage_input,
            label,
            per_stage_deadline_secs,
            engine,
        )?;
        let output_bounds = output_bounds_from_crown_result(&crown_result, &output_name, label)?;
        ensure_bounded_tensor_finite_and_ordered(&output_bounds, &format!("{label} output"))?;
        let provenance = stage_provenance(&crown_result);
        stage_input = output_bounds.clone();
        tightened_outputs.push(output_bounds);
        provenances.push(provenance);
    }

    let [x2_bounds, x3_bounds, x4_bounds]: [BoundedTensor; 3] = tightened_outputs
        .try_into()
        .map_err(|_| "expected three stage outputs".to_string())?;
    let [stage_a_prov, stage_b_prov, stage_c_prov]: [StageProvenance; 3] =
        provenances
            .try_into()
            .map_err(|_| "expected three stage provenance entries".to_string())?;

    let mut tightened_map = HashMap::new();
    tightened_map.insert(boundary.block_outputs[0].clone(), x2_bounds.clone());
    tightened_map.insert(boundary.block_outputs[1].clone(), x3_bounds.clone());
    tightened_map.insert(boundary.block_outputs[2].clone(), x4_bounds.clone());
    let mfa_bounds = concat_mfa_block_bounds(&boundary, &tightened_map);
    ensure_bounded_tensor_finite_and_ordered(&mfa_bounds, "stage-local MFA bounds")?;

    Ok(EcapaStageResult {
        x2_bounds,
        x3_bounds,
        x4_bounds,
        mfa_bounds,
        stage_provenances: [stage_a_prov, stage_b_prov, stage_c_prov],
    })
}

pub(crate) fn run_ecapa_stage_local_crown_ibp(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    per_stage_deadline_secs: u64,
) -> Result<EcapaStageResult, String> {
    run_ecapa_stage_local_crown_ibp_with_engine(graph, input, per_stage_deadline_secs, None)
}
