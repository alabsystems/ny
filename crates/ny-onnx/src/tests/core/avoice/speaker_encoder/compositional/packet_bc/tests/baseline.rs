// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::boundary::discover_ecapa_composition_boundary;
use super::support::assert_stage_outputs_contain_center;
use super::*;

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_stage_local_crown_ibp_preserves_or_tightens_stage_outputs_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary =
        discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery should succeed");
    let full_ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("full encoder IBP node-bound collection should succeed");
    let stage_result =
        run_ecapa_stage_local_crown_ibp(graph, &input, SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS)
            .expect("stage-local CROWN-IBP pipeline should succeed");

    assert_stage_outputs_contain_center(graph, &input, &boundary, &stage_result);

    let mut any_output_strict_improvement = false;
    for (label, output_name, tightened) in [
        ("x2", &boundary.block_outputs[0], &stage_result.x2_bounds),
        ("x3", &boundary.block_outputs[1], &stage_result.x3_bounds),
        ("x4", &boundary.block_outputs[2], &stage_result.x4_bounds),
    ] {
        let ibp_bounds = full_ibp_bounds
            .get(output_name)
            .unwrap_or_else(|| panic!("full encoder IBP missing stage output '{output_name}'"));
        assert_crown_tighter_than_ibp(tightened, ibp_bounds, label);
        let ibp_width = ibp_bounds.max_width();
        let tightened_width = tightened.max_width();
        let reduction_pct = width_reduction_pct(ibp_width, tightened_width);
        let tol = 1e-6 * ibp_width.max(tightened_width).max(1.0);
        any_output_strict_improvement |= tightened_width < ibp_width - tol;
        eprintln!(
            "{label}: full-graph IBP max_width {:.6} -> stage-local CROWN-IBP {:.6} ({reduction_pct:.1}% reduction)",
            ibp_width,
            tightened_width,
        );
    }

    assert!(
        stage_result
            .stage_provenances
            .iter()
            .any(|provenance| provenance.crown_ibp_tightened_count > 0),
        "expected at least one stage to tighten internal nodes with CROWN-IBP"
    );
    assert_finite_and_ordered(&stage_result.mfa_bounds, "stage-local MFA bounds");
    if !any_output_strict_improvement {
        eprintln!(
            "stage-local CROWN-IBP tightened internal nodes but all stage outputs \
             stayed at IBP parity; current output-level fallback remains tracked in #3680"
        );
    }
    eprintln!(
        "stage-local provenance: {:?}; any_output_strict_improvement={any_output_strict_improvement}",
        stage_result.stage_provenances,
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(900000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_compositional_cosine_bounds_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let result = run_ecapa_compositional_cosine_bounds(
        graph,
        &input,
        SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
    )
    .expect("compositional cosine pipeline should succeed");

    // Concrete-point containment (#3683): evaluate full encoder at center
    // and check the concrete output is within the full-graph IBP bounds.
    {
        let concrete = evaluate_graph_at_center(graph, &input, "compositional cosine encoder");
        assert_concrete_contained_in_bounds(
            &concrete,
            &graph
                .propagate_ibp(&input)
                .expect("full graph IBP should succeed"),
            "compositional cosine full-graph IBP containment",
        );
    }

    assert_finite_and_ordered(&result.stage_result.x2_bounds, "compositional x2 bounds");
    assert_finite_and_ordered(&result.stage_result.x3_bounds, "compositional x3 bounds");
    assert_finite_and_ordered(&result.stage_result.x4_bounds, "compositional x4 bounds");
    assert_finite_and_ordered(&result.stage_result.mfa_bounds, "compositional MFA bounds");
    assert!(
        result.distance_upper.is_finite(),
        "distance upper should be finite, got {}",
        result.distance_upper
    );
    assert!(
        result.dot_lower <= result.dot_upper,
        "dot bounds should be ordered: [{}, {}]",
        result.dot_lower,
        result.dot_upper
    );
    assert!(
        result.normsq_lower <= result.normsq_upper,
        "normsq bounds should be ordered: [{}, {}]",
        result.normsq_lower,
        result.normsq_upper
    );
    assert!(
        result.normsq_lower >= 0.0,
        "normsq lower should be non-negative, got {}",
        result.normsq_lower
    );
    eprintln!(
        "compositional cosine: dot=[{}, {}], normsq=[{}, {}], distance_upper={}, nonvacuous={}, stage_provenances={:?}",
        result.dot_lower,
        result.dot_upper,
        result.normsq_lower,
        result.normsq_upper,
        result.distance_upper,
        result.nonvacuous,
        result.stage_result.stage_provenances,
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(900000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_compositional_cosine_no_looser_than_monolithic_ibp_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let monolithic = run_monolithic_ibp_cosine_bounds(&input, SPEAKER_COMPONENT_SPEC_DEADLINE_SECS)
        .expect("monolithic cosine baseline should succeed");
    let compositional = run_ecapa_compositional_cosine_bounds(
        graph,
        &input,
        SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
    )
    .expect("compositional cosine pipeline should succeed");

    let monolithic_dot_width = scalar_width(monolithic.dot_lower, monolithic.dot_upper);
    let compositional_dot_width = scalar_width(compositional.dot_lower, compositional.dot_upper);
    let monolithic_normsq_width = scalar_width(monolithic.normsq_lower, monolithic.normsq_upper);
    let compositional_normsq_width =
        scalar_width(compositional.normsq_lower, compositional.normsq_upper);
    let dot_tol = 1e-6 * monolithic_dot_width.max(compositional_dot_width).max(1.0);
    let normsq_tol = 1e-6
        * monolithic_normsq_width
            .max(compositional_normsq_width)
            .max(1.0);

    assert!(
        compositional_dot_width <= monolithic_dot_width + dot_tol,
        "compositional dot width should be no worse than monolithic IBP: compositional={compositional_dot_width}, monolithic={monolithic_dot_width}"
    );
    assert!(
        compositional_normsq_width <= monolithic_normsq_width + normsq_tol,
        "compositional normsq width should be no worse than monolithic IBP: compositional={compositional_normsq_width}, monolithic={monolithic_normsq_width}"
    );

    eprintln!(
        "compositional vs monolithic: dot {:.6} -> {:.6} ({:.1}% reduction), normsq {:.6} -> {:.6} ({:.1}% reduction), distance {} -> {}",
        monolithic_dot_width,
        compositional_dot_width,
        width_reduction_pct(monolithic_dot_width, compositional_dot_width),
        monolithic_normsq_width,
        compositional_normsq_width,
        width_reduction_pct(monolithic_normsq_width, compositional_normsq_width),
        monolithic.distance_upper,
        compositional.distance_upper,
    );
    if !monolithic.nonvacuous {
        if compositional.nonvacuous {
            eprintln!(
                "compositional cosine achieved a nonvacuous bound while the monolithic IBP baseline remained vacuous"
            );
        } else {
            eprintln!(
                "both compositional and monolithic cosine bounds remain vacuous on the current ECAPA surface"
            );
        }
    }
}
