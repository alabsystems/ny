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

fn assert_and_log_output_comparison(
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
    assert!(
        ibp_out.0.is_finite()
            && ibp_out.1.is_finite()
            && cibp_out.0.is_finite()
            && cibp_out.1.is_finite(),
        "dot output comparison must stay finite: IBP={ibp_out:?}, CROWN-IBP={cibp_out:?}"
    );
    assert!(ibp_out.0 <= ibp_out.1 && cibp_out.0 <= cibp_out.1);
    let scale = ibp_out
        .0
        .abs()
        .max(ibp_out.1.abs())
        .max(cibp_out.0.abs())
        .max(cibp_out.1.abs())
        .max(1.0);
    let tol = 1e-4 * scale;
    assert!(
        cibp_out.0 >= ibp_out.0 - tol && cibp_out.1 <= ibp_out.1 + tol,
        "CROWN-IBP intermediates loosened the scalar dot enclosure: \
         IBP={ibp_out:?}, CROWN-IBP={cibp_out:?}"
    );
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

fn assert_complete_finite_no_looser_than_ibp(
    result: &ny_propagate::types::GraphCrownIbpBoundsResult,
    ibp_bounds: &HashMap<String, BoundedTensor>,
) {
    assert_eq!(
        result.bounds.len(),
        ibp_bounds.len(),
        "deadline fallback must still publish one bound for every IBP node"
    );
    assert_eq!(
        result.provenance.len(),
        result.bounds.len(),
        "every published CROWN-IBP bound must have provenance"
    );

    for (name, ibp) in ibp_bounds {
        let cibp = result
            .bounds
            .get(name)
            .unwrap_or_else(|| panic!("CROWN-IBP result omitted node {name}"));
        assert!(
            result.provenance.contains_key(name),
            "CROWN-IBP result omitted provenance for node {name}"
        );
        assert_eq!(
            cibp.shape(),
            ibp.shape(),
            "CROWN-IBP changed the bound shape for node {name}"
        );
        for (index, (((&ibp_l, &ibp_u), &cibp_l), &cibp_u)) in ibp
            .lower()
            .iter()
            .zip(ibp.upper().iter())
            .zip(cibp.lower().iter())
            .zip(cibp.upper().iter())
            .enumerate()
        {
            assert!(
                ibp_l.is_finite() && ibp_u.is_finite() && cibp_l.is_finite() && cibp_u.is_finite(),
                "node {name}[{index}] produced non-finite bounds: \
                 IBP=[{ibp_l}, {ibp_u}], CROWN-IBP=[{cibp_l}, {cibp_u}]"
            );
            assert!(
                ibp_l <= ibp_u && cibp_l <= cibp_u,
                "node {name}[{index}] produced inverted bounds"
            );
            let scale = ibp_l
                .abs()
                .max(ibp_u.abs())
                .max(cibp_l.abs())
                .max(cibp_u.abs())
                .max(1.0);
            let tol = 1e-4 * scale;
            assert!(
                cibp_l >= ibp_l - tol && cibp_u <= ibp_u + tol,
                "node {name}[{index}] CROWN-IBP loosened IBP: \
                 IBP=[{ibp_l}, {ibp_u}], CROWN-IBP=[{cibp_l}, {cibp_u}]"
            );
        }
    }
}

/// A deadline-bounded CROWN-IBP pass must preserve complete, finite, no-looser
/// node bounds even when individual nodes fall back. Provenance and timing are
/// logged to diagnose how much real CROWN work fit in the 60s budget. #3596
///
/// Uses the same deadline as the comparison test to diagnose the provenance
/// distribution: how many nodes are CROWN-tightened, how many fall back to
/// IBP (deadline, memory, error), and what reduction each tightened node achieves.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_crown_ibp_dag_deadline_fallback_is_complete_and_sound_3596() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
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

    assert_complete_finite_no_looser_than_ibp(&cibp_result, &ibp_bounds);
    log_provenance_summary(
        &cibp_result,
        &ibp_bounds,
        shared::SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
        t0.elapsed(),
    );
    assert_and_log_output_comparison(&dot_graph, &input, &ibp_bounds, &cibp_result.bounds);
}
