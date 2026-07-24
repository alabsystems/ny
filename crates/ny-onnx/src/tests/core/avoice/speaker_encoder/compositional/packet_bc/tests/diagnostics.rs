// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::boundary::discover_ecapa_composition_boundary;
use super::*;

/// Log the layer-type inventory of a graph subgraph.
fn log_layer_type_inventory(graph: &GraphNetwork, label: &str) {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            *counts
                .entry(node.layer().layer_type().to_string())
                .or_insert(0) += 1;
        }
    }
    eprintln!(
        "DIAG {label}: {} nodes, output='{}'",
        graph.num_nodes(),
        graph.output_name()
    );
    for (layer_type, count) in &counts {
        eprintln!("  {layer_type}: {count}");
    }
}

/// Log all fallback events and per-node fallback provenance for a CROWN-IBP result.
fn log_all_crown_ibp_fallbacks(
    label: &str,
    graph: &GraphNetwork,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    crown_result: &GraphCrownIbpBoundsResult,
) {
    for (idx, event) in crown_result.fallback_events.iter().enumerate() {
        eprintln!(
            "DIAG {label} fallback[{idx}]: layer_idx={}, layer_type={}, reason={:?}, details={}",
            event.layer_index, event.layer_type, event.reason, event.details
        );
    }
    let topo = graph.topological_sort().unwrap();
    for node_name in &topo {
        if let Some(prov) = crown_result.provenance.get(node_name) {
            if prov.is_fallback() {
                let node = graph.node(node_name).unwrap();
                let ibp_w = ibp_bounds
                    .get(node_name)
                    .map(|b| b.max_width())
                    .unwrap_or(f32::NAN);
                let cibp_w = crown_result
                    .bounds
                    .get(node_name)
                    .map(|b| b.max_width())
                    .unwrap_or(f32::NAN);
                eprintln!(
                    "DIAG {label} fallback_node: '{}' layer_type={} inputs={:?} ibp={:.6} cibp={:.6} prov={:?}",
                    node_name, node.layer().layer_type(), node.inputs(), ibp_w, cibp_w, prov,
                );
            }
        }
    }
}

/// Level 0 diagnostic for #3499: identify which node/op in Stage A triggers
/// CROWN-IBP fallback during the tightening loop.
///
/// Reference: `designs/2026-03-13-issue-3499-res2net-bound-widening-root-cause.md`
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_ecapa_stage_a_crown_ibp_fallback_diagnostic_3499() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary =
        discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery should succeed");
    let [stage_a, _stage_b, _stage_c] = extract_ecapa_stage_graphs(graph, &boundary)
        .expect("stage graph extraction should succeed");

    log_layer_type_inventory(&stage_a, "stage_a");

    let ibp_bounds = stage_a
        .collect_node_bounds(&input)
        .expect("Stage A IBP node-bound collection should succeed");
    let output_name = stage_a.output_name().to_string();
    let ibp_output = ibp_bounds
        .get(&output_name)
        .expect("Stage A IBP missing output node");
    eprintln!(
        "DIAG stage_a: IBP ({} nodes) in {:.1}s, output max_width={:.6}, shape={:?}",
        ibp_bounds.len(),
        t_start.elapsed().as_secs_f64(),
        ibp_output.max_width(),
        ibp_output.lower().shape(),
    );

    let deadline = Instant::now() + Duration::from_mins(2);
    let crown_result = stage_a
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &input,
            ibp_bounds.clone(),
            Some(deadline),
        )
        .expect("Stage A CROWN-IBP collection should succeed");

    let crown_count = crown_result
        .provenance
        .values()
        .filter(|p| matches!(p, BoundsProvenance::Crown))
        .count();
    let fallback_count = crown_result
        .provenance
        .values()
        .filter(|p| p.is_fallback())
        .count();
    eprintln!(
        "DIAG stage_a: CROWN-IBP in {:.1}s, {}/{} tightened, {} fallback",
        t_start.elapsed().as_secs_f64(),
        crown_count,
        crown_result.bounds.len(),
        fallback_count,
    );

    log_all_crown_ibp_fallbacks("stage_a", &stage_a, &ibp_bounds, &crown_result);

    let crown_output = crown_result.bounds.get(&output_name).unwrap();
    eprintln!(
        "DIAG stage_a: output prov={:?}, crown_w={:.6}, ibp_w={:.6}",
        crown_result.provenance_for_node(&output_name),
        crown_output.max_width(),
        ibp_output.max_width(),
    );
    assert_finite_and_ordered(crown_output, "Stage A CROWN-IBP output");
}

/// Regression for #3499/#3718: the ECAPA stage-local tightening helper must
/// thread a GemmEngine into Conv1d CROWN backward instead of staying on the
/// default CPU-only collector path.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_ecapa_stage_a_crown_ibp_uses_gemm_engine_3499() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    use ny_test_utils::CountingGemmEngine;

    const ENGINE_TEST_STAGE_DEADLINE_SECS: u64 = 20;

    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary =
        discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery should succeed");
    let [stage_a, _stage_b, _stage_c] = extract_ecapa_stage_graphs(graph, &boundary)
        .expect("stage graph extraction should succeed");
    let engine = CountingGemmEngine::new();

    let crown_result = collect_ecapa_stage_local_crown_ibp(
        &stage_a,
        &input,
        "stage_a_engine",
        ENGINE_TEST_STAGE_DEADLINE_SECS,
        Some(&engine),
    )
    .expect("Stage A engine-aware CROWN-IBP collection should succeed");
    let gemm_count = engine.gemm_calls();
    let output_bounds = output_bounds_from_crown_result(
        &crown_result,
        stage_a.output_name(),
        "stage_a engine-aware output",
    )
    .expect("Stage A engine-aware output bounds should exist");

    assert!(
        gemm_count > 0,
        "GemmEngine received 0 GEMM calls — Stage A CROWN-IBP is not threading engine through Conv1d backward"
    );
    assert_finite_and_ordered(&output_bounds, "Stage A engine-aware CROWN-IBP output");
    eprintln!(
        "Stage A engine-aware CROWN-IBP dispatched {gemm_count} GEMM calls with a {}s deadline",
        ENGINE_TEST_STAGE_DEADLINE_SECS
    );
}
