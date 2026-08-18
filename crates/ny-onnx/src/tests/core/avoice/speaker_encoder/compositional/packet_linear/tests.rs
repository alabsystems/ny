// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::discover_ecapa_composition_boundary;
use super::super::packet_bc::core::{
    total_bound_width, width_reduction_pct, EcapaCosineResult, EcapaStageResult, StageProvenance,
};
use super::super::packet_bc::cosine::{
    cosine_bounds_from_stage_result, run_ecapa_compositional_cosine_bounds,
};
use super::super::packet_bc::stage_local::run_ecapa_stage_local_crown_ibp;
use super::composition::cosine_bounds_with_linear_composition;
use super::*;

mod packet2;

/// Per-prefix CROWN-IBP tightening budget (seconds) for the direct-boundary
/// approach. The identity-spec CROWN backward runs without a deadline because
/// LinearBounds extraction is all-or-nothing: any deadline-triggered IBP
/// fallback causes the entire backward pass to return None for LinearBounds.
/// Overall time is bounded by the ntest timeout.
const DIRECT_BOUNDARY_CROWN_IBP_BUDGET_SECS: u64 = 30;

/// Verify that raw-input prefix graphs can be extracted for all three ECAPA
/// block outputs, rooted at NETWORK_INPUT rather than at the preceding stage.
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_raw_input_prefix_graphs_extracted_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let graph = avoice_speaker_encoder_graph();
    let boundary =
        discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery should succeed");
    let [prefix_x2, prefix_x3, prefix_x4] = extract_ecapa_raw_input_prefix_graphs(graph, &boundary)
        .expect("raw-input prefix graph extraction should succeed");

    assert!(prefix_x2.num_nodes() > 0, "prefix_x2 should have nodes");
    assert!(
        prefix_x3.num_nodes() >= prefix_x2.num_nodes(),
        "prefix_x3 >= prefix_x2: {} vs {}",
        prefix_x3.num_nodes(),
        prefix_x2.num_nodes()
    );
    assert!(
        prefix_x4.num_nodes() >= prefix_x3.num_nodes(),
        "prefix_x4 >= prefix_x3: {} vs {}",
        prefix_x4.num_nodes(),
        prefix_x3.num_nodes()
    );

    for (label, prefix) in [
        ("prefix_x2", &prefix_x2),
        ("prefix_x3", &prefix_x3),
        ("prefix_x4", &prefix_x4),
    ] {
        assert!(
            prefix.topological_sort().is_ok(),
            "{label}: topological sort should succeed"
        );
        eprintln!(
            "{label}: {} nodes, output='{}'",
            prefix.num_nodes(),
            prefix.output_name()
        );
    }
}

/// Core regression for the direct-boundary alternative design (#3499):
/// the MFA bounds from direct raw-input certificates should be strictly no
/// looser than the stage-local interval-chained MFA bounds.
///
/// Reference: designs/2026-03-13-issue-3499-linear-boundary-certificates-alternative.md
/// Acceptance criterion 1.
#[cfg_attr(not(debug_assertions), ntest::timeout(1800000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_direct_boundary_mfa_no_looser_than_stage_local_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );

    let stage_result =
        run_ecapa_stage_local_crown_ibp(graph, &input, SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS)
            .expect("stage-local CROWN-IBP should succeed");
    let stage_mfa_max_w = stage_result.mfa_bounds.max_width();
    let stage_mfa_total_w = total_bound_width(&stage_result.mfa_bounds);

    let direct_result =
        run_ecapa_direct_boundary_mfa_bounds(graph, &input, DIRECT_BOUNDARY_CROWN_IBP_BUDGET_SECS)
            .expect("direct-boundary MFA bounds should succeed");
    let direct_mfa_max_w = direct_result.mfa_bounds.max_width();
    let direct_mfa_total_w = total_bound_width(&direct_result.mfa_bounds);
    eprintln!(
        "direct-boundary: prefix_dims={:?}, total_mfa_dim={}, linear_path={}",
        direct_result.prefix_output_dims,
        direct_result.total_mfa_dim,
        direct_result.used_linear_path,
    );
    // Gate: the tightening comparison is only meaningful on the linear path.
    // Concrete fallback produces IBP-quality bounds that may be at parity
    // or worse than stage-local — silently passing would hide a regression.
    assert!(
        direct_result.used_linear_path,
        "direct-boundary should use linear certificate path, not concrete fallback"
    );

    let max_tol = 1e-5 * stage_mfa_max_w.max(direct_mfa_max_w).max(1.0);
    let total_tol = 1e-5 * stage_mfa_total_w.max(direct_mfa_total_w).max(1.0);
    eprintln!(
        "MFA max_width: stage={:.6}, direct={:.6} ({:.1}%)",
        stage_mfa_max_w,
        direct_mfa_max_w,
        width_reduction_pct(stage_mfa_max_w, direct_mfa_max_w),
    );
    eprintln!(
        "MFA total_width: stage={:.6}, direct={:.6} ({:.1}%)",
        stage_mfa_total_w,
        direct_mfa_total_w,
        width_reduction_pct(stage_mfa_total_w, direct_mfa_total_w),
    );
    assert!(
        direct_mfa_max_w <= stage_mfa_max_w + max_tol,
        "direct max width no worse: {direct_mfa_max_w} vs {stage_mfa_max_w}"
    );
    assert!(
        direct_mfa_total_w <= stage_mfa_total_w + total_tol,
        "direct total width no worse: {direct_mfa_total_w} vs {stage_mfa_total_w}"
    );

    let strict = direct_mfa_total_w < stage_mfa_total_w - total_tol;
    if strict {
        eprintln!(
            "direct-boundary STRICTLY TIGHTER ({:.1}% reduction)",
            width_reduction_pct(stage_mfa_total_w, direct_mfa_total_w),
        );
    } else {
        eprintln!("direct-boundary at parity (tolerance {total_tol:.6})");
    }
}

/// Build a stub EcapaStageResult wrapping the given MFA bounds for the
/// cosine suffix pipeline.
fn stub_stage_result_for_mfa(mfa_bounds: BoundedTensor) -> EcapaStageResult {
    let stub_prov = StageProvenance {
        node_count: 0,
        crown_ibp_tightened_count: 0,
        ibp_fallback_count: 0,
    };
    EcapaStageResult {
        x2_bounds: BoundedTensor::new_conservative(&[1]),
        x3_bounds: BoundedTensor::new_conservative(&[1]),
        x4_bounds: BoundedTensor::new_conservative(&[1]),
        mfa_bounds,
        stage_provenances: [stub_prov; 3],
    }
}

/// Compare cosine bounds from direct-boundary vs stage-local, asserting
/// direct is no worse. Returns `(nonvacuous_direct, nonvacuous_stage)`.
fn assert_cosine_no_worse(
    stage: &EcapaCosineResult,
    direct_dot: (f32, f32),
    direct_normsq: (f32, f32),
    direct_distance: f32,
    direct_nonvacuous: bool,
) {
    let s_dot_w = scalar_width(stage.dot_lower, stage.dot_upper);
    let d_dot_w = scalar_width(direct_dot.0, direct_dot.1);
    let s_nrm_w = scalar_width(stage.normsq_lower, stage.normsq_upper);
    let d_nrm_w = scalar_width(direct_normsq.0, direct_normsq.1);

    eprintln!(
        "cosine: stage dot_w={s_dot_w:.6} direct={d_dot_w:.6}, \
         stage normsq_w={s_nrm_w:.6} direct={d_nrm_w:.6}"
    );
    eprintln!(
        "cosine: stage dist={}, direct dist={direct_distance}",
        stage.distance_upper,
    );
    eprintln!(
        "cosine: stage nonvac={}, direct nonvac={direct_nonvacuous}",
        stage.nonvacuous,
    );

    let dot_tol = 1e-4 * s_dot_w.max(d_dot_w).max(1.0);
    let nrm_tol = 1e-4 * s_nrm_w.max(d_nrm_w).max(1.0);
    assert!(
        d_dot_w <= s_dot_w + dot_tol,
        "direct dot no worse: {d_dot_w} vs {s_dot_w}"
    );
    assert!(
        d_nrm_w <= s_nrm_w + nrm_tol,
        "direct normsq no worse: {d_nrm_w} vs {s_nrm_w}"
    );
}

/// End-to-end: run the direct-boundary MFA certificate through the existing
/// cosine suffix pipeline and compare against the stage-local cosine result.
///
/// Acceptance criterion 2: the direct-boundary cosine result should be
/// no worse than the current interval-stage cosine packet.
#[cfg_attr(not(debug_assertions), ntest::timeout(1800000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_direct_boundary_cosine_no_worse_than_stage_local_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );

    let stage_cosine = run_ecapa_compositional_cosine_bounds(
        graph,
        &input,
        SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
    )
    .expect("stage-local cosine should succeed");

    let direct_mfa =
        run_ecapa_direct_boundary_mfa_bounds(graph, &input, DIRECT_BOUNDARY_CROWN_IBP_BUDGET_SECS)
            .expect("direct-boundary MFA should succeed");

    // Gate: cosine comparison only meaningful on linear path (see AC1 comment).
    assert!(
        direct_mfa.used_linear_path,
        "direct-boundary should use linear certificate path for cosine comparison"
    );

    let direct_stage = stub_stage_result_for_mfa(direct_mfa.mfa_bounds);
    let (dot_l, dot_u, nrm_l, nrm_u) = cosine_bounds_from_stage_result(
        &direct_stage,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
        "direct",
    )
    .expect("direct-boundary cosine suffix should succeed");
    let (dist_upper, nonvacuous) = speaker_cosine_distance_upper(dot_l, nrm_u);

    assert_cosine_no_worse(
        &stage_cosine,
        (dot_l, dot_u),
        (nrm_l, nrm_u),
        dist_upper,
        nonvacuous,
    );

    if nonvacuous && !stage_cosine.nonvacuous {
        eprintln!("direct-boundary cosine NONVACUOUS while stage-local vacuous");
    } else if !nonvacuous {
        eprintln!("direct-boundary cosine still vacuous — next: linear suffix extension");
    }
}
