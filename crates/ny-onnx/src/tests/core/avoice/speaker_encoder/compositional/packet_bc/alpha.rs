// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::discover_ecapa_composition_boundary;
use super::super::subgraph::concat_mfa_block_bounds;
use super::core::{
    ensure_bounded_tensor_finite_and_ordered, extract_ecapa_stage_graphs, output_bounds_from_map,
    EcapaStageResult, StageProvenance,
};
use super::*;

pub(crate) fn alpha_crown_config_for_stage(deadline_secs: u64) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: false,
        sparse_ratio: 0.2,
        deadline: Some(Instant::now() + Duration::from_secs(deadline_secs)),
        ..AlphaCrownConfig::default()
    }
}

pub(crate) fn alpha_crown_stage_deadline_budget(config: &AlphaCrownConfig) -> Option<Duration> {
    config
        .deadline
        .map(|deadline| deadline.saturating_duration_since(Instant::now()))
}

pub(crate) fn refreshed_alpha_crown_stage_config(
    config: &AlphaCrownConfig,
    stage_budget: Option<Duration>,
) -> AlphaCrownConfig {
    let mut stage_config = config.clone();
    stage_config.deadline = stage_budget.map(|budget| Instant::now() + budget);
    stage_config
}

pub(crate) fn run_ecapa_stage_local_alpha_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &AlphaCrownConfig,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> Result<EcapaStageResult, String> {
    let boundary = discover_ecapa_composition_boundary(graph)?;
    let [stage_a, stage_b, stage_c] = extract_ecapa_stage_graphs(graph, &boundary)?;
    let stage_deadline_budget = alpha_crown_stage_deadline_budget(config);
    let mut stage_input = input.clone();
    let mut tightened_outputs = Vec::with_capacity(3);
    let mut provenances = Vec::with_capacity(3);

    for (label, stage) in [
        ("alpha_stage_a", &stage_a),
        ("alpha_stage_b", &stage_b),
        ("alpha_stage_c", &stage_c),
    ] {
        let stage_config = refreshed_alpha_crown_stage_config(config, stage_deadline_budget);
        let output_name = stage.output_name().to_string();
        let (alpha_bounds, _alpha_state) = stage
            .collect_alpha_crown_bounds_dag_with_engine(&stage_input, &stage_config, engine)
            .map_err(|e| format!("{label}: alpha-CROWN failed: {e}"))?;
        let output_bounds = output_bounds_from_map(&alpha_bounds, &output_name, label)?;
        ensure_bounded_tensor_finite_and_ordered(&output_bounds, &format!("{label} output"))?;
        let prov = StageProvenance {
            node_count: alpha_bounds.len(),
            crown_ibp_tightened_count: alpha_bounds.len(),
            ibp_fallback_count: 0,
        };
        eprintln!(
            "{label}: nodes={}, alpha-CROWN max_width={:.6}",
            prov.node_count,
            output_bounds.max_width(),
        );
        stage_input = output_bounds.clone();
        tightened_outputs.push(output_bounds);
        provenances.push(prov);
    }

    let [x2_bounds, x3_bounds, x4_bounds]: [BoundedTensor; 3] = tightened_outputs
        .try_into()
        .map_err(|_| "expected three stage outputs".to_string())?;
    let [prov_a, prov_b, prov_c]: [StageProvenance; 3] = provenances
        .try_into()
        .map_err(|_| "expected three provenances".to_string())?;

    let mut tightened_map = HashMap::new();
    tightened_map.insert(boundary.block_outputs[0].clone(), x2_bounds.clone());
    tightened_map.insert(boundary.block_outputs[1].clone(), x3_bounds.clone());
    tightened_map.insert(boundary.block_outputs[2].clone(), x4_bounds.clone());
    let mfa_bounds = concat_mfa_block_bounds(&boundary, &tightened_map);
    ensure_bounded_tensor_finite_and_ordered(&mfa_bounds, "alpha stage-local MFA")?;

    Ok(EcapaStageResult {
        x2_bounds,
        x3_bounds,
        x4_bounds,
        mfa_bounds,
        stage_provenances: [prov_a, prov_b, prov_c],
    })
}
