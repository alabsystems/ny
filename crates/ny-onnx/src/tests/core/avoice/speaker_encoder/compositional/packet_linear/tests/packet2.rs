// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct Packet2Comparison {
    std_dot_w: f32,
    cmp_dot_w: f32,
    std_nrm_w: f32,
    cmp_nrm_w: f32,
    std_dist: f32,
    cmp_dist: f32,
    std_nonvac: bool,
    cmp_nonvac: bool,
}

fn run_packet2_comparison() -> Packet2Comparison {
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let direct_mfa =
        run_ecapa_direct_boundary_mfa_bounds(graph, &input, DIRECT_BOUNDARY_CROWN_IBP_BUDGET_SECS)
            .expect("direct-boundary MFA should succeed");
    assert!(
        direct_mfa.used_linear_path,
        "composed test requires linear certificate path"
    );
    let mfa_linear = direct_mfa
        .mfa_linear
        .as_ref()
        .expect("mfa_linear should be Some when used_linear_path is true");
    let std_stage = stub_stage_result_for_mfa(direct_mfa.mfa_bounds.clone());
    let (std_dot_l, std_dot_u, std_nrm_l, std_nrm_u) =
        cosine_bounds_from_stage_result(&std_stage, SPEAKER_COMPONENT_SPEC_DEADLINE_SECS, "std")
            .expect("standard cosine suffix should succeed");
    let (std_dist, std_nonvac) = speaker_cosine_distance_upper(std_dot_l, std_nrm_u);
    let (cmp_dot_l, cmp_dot_u, cmp_nrm_l, cmp_nrm_u) = cosine_bounds_with_linear_composition(
        mfa_linear,
        &input,
        &direct_mfa.mfa_bounds,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
        "composed",
    )
    .expect("composed cosine suffix should succeed");
    let (cmp_dist, cmp_nonvac) = speaker_cosine_distance_upper(cmp_dot_l, cmp_nrm_u);

    Packet2Comparison {
        std_dot_w: scalar_width(std_dot_l, std_dot_u),
        cmp_dot_w: scalar_width(cmp_dot_l, cmp_dot_u),
        std_nrm_w: scalar_width(std_nrm_l, std_nrm_u),
        cmp_nrm_w: scalar_width(cmp_nrm_l, cmp_nrm_u),
        std_dist,
        cmp_dist,
        std_nonvac,
        cmp_nonvac,
    }
}

fn log_packet2_comparison(comparison: &Packet2Comparison) {
    eprintln!("Packet 2 comparison:");
    eprintln!(
        "  dot width: std={:.6} composed={:.6} ({:.1}% reduction)",
        comparison.std_dot_w,
        comparison.cmp_dot_w,
        width_reduction_pct(comparison.std_dot_w, comparison.cmp_dot_w),
    );
    eprintln!(
        "  normsq width: std={:.6} composed={:.6} ({:.1}% reduction)",
        comparison.std_nrm_w,
        comparison.cmp_nrm_w,
        width_reduction_pct(comparison.std_nrm_w, comparison.cmp_nrm_w),
    );
    eprintln!(
        "  distance: std={} composed={}",
        comparison.std_dist, comparison.cmp_dist
    );
    eprintln!(
        "  nonvacuous: std={} composed={}",
        comparison.std_nonvac, comparison.cmp_nonvac
    );
}

/// Packet 2: compose suffix LinearBounds with prefix MFA LinearBounds and
/// intersect with standard (concretize-first) bounds for best-of-both (#3499).
///
/// The composition pipeline:
/// 1. Gets MFA LinearBounds from the direct-boundary prefix packet
/// 2. Runs suffix CROWN requesting LinearBounds over MFA
/// 3. Composes suffix LinearBounds with prefix MFA LinearBounds
/// 4. Concretizes the composed bounds on the raw input domain
/// 5. Intersects composed with standard bounds (max of lowers, min of uppers)
///
/// The intersection is always no worse than standard (both are valid
/// over-approximations). Composition helps when cross-dimensional MFA
/// correlation outweighs the McCormick decomposition looseness.
///
/// Reference: designs/2026-03-13-issue-3499-suffix-linear-certificate-composition.md
#[cfg_attr(not(debug_assertions), ntest::timeout(1800000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_composed_cosine_no_worse_than_concretized_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let comparison = run_packet2_comparison();
    log_packet2_comparison(&comparison);

    let dot_tol = 1e-4 * comparison.std_dot_w.max(comparison.cmp_dot_w).max(1.0);
    let nrm_tol = 1e-4 * comparison.std_nrm_w.max(comparison.cmp_nrm_w).max(1.0);
    assert!(
        comparison.cmp_dot_w <= comparison.std_dot_w + dot_tol,
        "composed dot width no worse: {} vs {}",
        comparison.cmp_dot_w,
        comparison.std_dot_w
    );
    assert!(
        comparison.cmp_nrm_w <= comparison.std_nrm_w + nrm_tol,
        "composed normsq width no worse: {} vs {}",
        comparison.cmp_nrm_w,
        comparison.std_nrm_w
    );

    if comparison.cmp_nonvac && !comparison.std_nonvac {
        eprintln!(
            "BREAKTHROUGH: composed cosine NONVACUOUS (distance_upper={}) \
             while standard path vacuous",
            comparison.cmp_dist
        );
    } else if comparison.cmp_nonvac {
        eprintln!(
            "composed cosine nonvacuous: distance_upper={} (std={})",
            comparison.cmp_dist, comparison.std_dist
        );
    } else {
        eprintln!(
            "composed cosine still vacuous — suffix nonlinearity dominates; \
             next step: alpha-CROWN suffix optimization (Packet 3)"
        );
    }
}
