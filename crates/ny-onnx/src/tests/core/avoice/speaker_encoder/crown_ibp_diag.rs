// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::cosine_head::{scalar_spec_bounds_with_node_bounds, scalar_width};
use super::*;

fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum()
}

fn log_crown_tightened_nodes(
    ibp_bounds: &HashMap<String, BoundedTensor>,
    cibp_bounds: &HashMap<String, BoundedTensor>,
    crown_tightened: &[&String],
) {
    for name in crown_tightened {
        if let (Some(ibp), Some(cibp)) = (ibp_bounds.get(*name), cibp_bounds.get(*name)) {
            let ibp_width = total_bound_width(ibp);
            let cibp_width = total_bound_width(cibp);
            let reduction_pct = if ibp_width > 0.0 {
                (1.0 - cibp_width / ibp_width) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  CROWN: {} dim={} ibp_total_width={:.6} cibp_total_width={:.6} reduction={:.2}%",
                name,
                ibp.len(),
                ibp_width,
                cibp_width,
                reduction_pct
            );
        }
    }
}

fn log_fallback_details(
    fallback_nodes: &[(&String, &BoundsProvenance)],
    fallback_events: &[ny_propagate::types::CrownIbpFallbackEvent],
) {
    for (name, prov) in fallback_nodes {
        eprintln!("  FALLBACK: {} reason={:?}", name, prov);
    }
    for event in fallback_events {
        eprintln!(
            "  EVENT[{}]: layer_type={} reason={:?} details={}",
            event.layer_index, event.layer_type, event.reason, event.details
        );
    }
}

fn log_provenance_summary(
    result: &ny_propagate::types::GraphCrownIbpBoundsResult,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    deadline_secs: u64,
    elapsed: Duration,
) {
    let total_nodes = result.bounds.len();
    let crown_tightened: Vec<&String> = result
        .provenance
        .iter()
        .filter(|(_, prov)| matches!(prov, BoundsProvenance::Crown))
        .map(|(name, _)| name)
        .collect();
    let fallback_nodes: Vec<(&String, &BoundsProvenance)> = result
        .provenance
        .iter()
        .filter(|(_, prov)| !matches!(prov, BoundsProvenance::Crown))
        .collect();
    eprintln!(
        "CROWN-IBP DAG provenance (dot, {}s deadline): {}/{} nodes CROWN-tightened, {} fell back, took {:.1}s",
        deadline_secs, crown_tightened.len(), total_nodes, fallback_nodes.len(), elapsed.as_secs_f64()
    );
    log_crown_tightened_nodes(ibp_bounds, &result.bounds, &crown_tightened);
    log_fallback_details(&fallback_nodes, &result.fallback_events);
}

fn log_output_comparison(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    cibp_bounds: &HashMap<String, BoundedTensor>,
) {
    let ibp_out = scalar_spec_bounds_with_node_bounds(graph, input, ibp_bounds, "diag dot + IBP");
    let cibp_out =
        scalar_spec_bounds_with_node_bounds(graph, input, cibp_bounds, "diag dot + CROWN-IBP");
    let ibp_width = scalar_width(ibp_out.0, ibp_out.1);
    let cibp_width = scalar_width(cibp_out.0, cibp_out.1);
    eprintln!(
        "dot output: ibp=[{}, {}] width={}; cibp=[{}, {}] width={}; improved={}",
        ibp_out.0,
        ibp_out.1,
        ibp_width,
        cibp_out.0,
        cibp_out.1,
        cibp_width,
        cibp_width < ibp_width
    );
}

/// Diagnostic: report which nodes get CROWN-IBP tightened and which fall back
/// on the dot component graph with a 60s deadline (#3596).
///
/// Uses the same deadline as the comparison test to diagnose the provenance
/// distribution: how many nodes are CROWN-tightened, how many fall back to
/// IBP (deadline, memory, error), and what reduction each tightened node achieves.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_speaker_crown_ibp_dag_provenance_diagnostic_3596() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let (dot_graph, _, _) = cosine_head::build_speaker_cosine_component_graphs();
    eprintln!(
        "DIAG: graph built in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );

    let model = shared::avoice_speaker_encoder();
    let input = shared::bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );
    eprintln!(
        "DIAG: input created in {:.1}s total",
        t_start.elapsed().as_secs_f64()
    );

    let ibp_bounds = dot_graph
        .collect_node_bounds(&input)
        .expect("IBP collection should succeed");
    eprintln!(
        "DIAG: IBP forward ({} nodes) in {:.1}s total",
        ibp_bounds.len(),
        t_start.elapsed().as_secs_f64()
    );

    // Use pre-computed IBP bounds to avoid the redundant forward pass that
    // previously consumed the entire deadline budget (#3596).
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs(shared::SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS);
    eprintln!(
        "DIAG: starting CROWN-IBP with {}s deadline",
        shared::SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS
    );
    let cibp_result = dot_graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &input,
            ibp_bounds.clone(),
            Some(deadline),
        )
        .expect("CROWN-IBP status collection should succeed");

    log_provenance_summary(
        &cibp_result,
        &ibp_bounds,
        shared::SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
        t0.elapsed(),
    );
    log_output_comparison(&dot_graph, &input, &ibp_bounds, &cibp_result.bounds);
}
