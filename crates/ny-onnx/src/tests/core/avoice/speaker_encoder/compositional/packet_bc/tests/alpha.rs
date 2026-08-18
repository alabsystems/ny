// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::boundary::discover_ecapa_composition_boundary;
use super::support::assert_stage_outputs_contain_center;
use super::*;

/// Regression for #3499: stage-local alpha-CROWN must refresh each extracted
/// stage's deadline budget instead of reusing one stale absolute `Instant`.
#[test]
fn test_ecapa_stage_local_alpha_crown_refreshes_per_stage_deadline_3499() {
    let config = alpha_crown_config_for_stage(1);
    let stage_budget = alpha_crown_stage_deadline_budget(&config)
        .expect("stage-local alpha config should include a deadline budget");

    std::thread::sleep(Duration::from_millis(25));
    let stage_a = refreshed_alpha_crown_stage_config(&config, Some(stage_budget));
    std::thread::sleep(Duration::from_millis(25));
    let stage_b = refreshed_alpha_crown_stage_config(&config, Some(stage_budget));

    let stage_a_deadline = stage_a
        .deadline
        .expect("refreshed stage A config should carry a deadline");
    let stage_b_deadline = stage_b
        .deadline
        .expect("refreshed stage B config should carry a deadline");

    assert!(
        stage_b_deadline > stage_a_deadline,
        "per-stage alpha deadlines should be refreshed instead of reusing one stale absolute Instant"
    );
    assert!(
        stage_a_deadline.saturating_duration_since(Instant::now())
            >= stage_budget.saturating_sub(Duration::from_millis(150)),
        "stage A should receive nearly the full per-stage alpha budget"
    );
    assert!(
        stage_b_deadline.saturating_duration_since(Instant::now())
            >= stage_budget.saturating_sub(Duration::from_millis(150)),
        "stage B should receive nearly the full per-stage alpha budget"
    );
}

const ALPHA_SMOKE_STAGE_DEADLINE_SECS: u64 = 20;
const ALPHA_SMOKE_ITERATIONS: usize = 1;

fn alpha_smoke_config_for_stage() -> AlphaCrownConfig {
    let mut config = alpha_crown_config_for_stage(ALPHA_SMOKE_STAGE_DEADLINE_SECS);
    config.iterations = ALPHA_SMOKE_ITERATIONS;
    config
}

/// Alpha-CROWN stage-local: optimize ReLU slopes on each ~40-60 node stage
/// subgraph. Compare the resulting stage outputs against the cheap full-graph
/// IBP baseline instead of rerunning the entire stage-local CROWN-IBP pipeline.
/// The dedicated CROWN-IBP comparison already lives in
/// `test_ecapa_stage_local_crown_ibp_preserves_or_tightens_stage_outputs_3499`;
/// duplicating it here pushes the alpha lane over the 600s test budget. Use a
/// small alpha smoke budget here; the long-budget quality measurements belong in
/// manual `#3499` runs, not in the regression surface.
/// Part of #3499 criterion 3.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_stage_local_alpha_crown_tightens_stage_outputs_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary = discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery");
    let full_ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("full encoder IBP node-bound collection should succeed");
    let config = alpha_smoke_config_for_stage();
    let alpha_start = Instant::now();
    let alpha_result = run_ecapa_stage_local_alpha_crown(graph, &input, &config, None)
        .expect("alpha-CROWN stage-local pipeline should succeed");
    let alpha_elapsed = alpha_start.elapsed().as_secs_f64();

    assert_stage_outputs_contain_center(graph, &input, &boundary, &alpha_result);
    assert_finite_and_ordered(&alpha_result.mfa_bounds, "alpha stage-local MFA");

    for (label, output_name, alpha_bounds) in [
        ("x2", &boundary.block_outputs[0], &alpha_result.x2_bounds),
        ("x3", &boundary.block_outputs[1], &alpha_result.x3_bounds),
        ("x4", &boundary.block_outputs[2], &alpha_result.x4_bounds),
    ] {
        let ibp_bounds = full_ibp_bounds
            .get(output_name)
            .unwrap_or_else(|| panic!("full encoder IBP missing stage output '{output_name}'"));
        assert_crown_tighter_than_ibp(alpha_bounds, ibp_bounds, label);
        let alpha_width = alpha_bounds.max_width();
        let ibp_width = ibp_bounds.max_width();
        let reduction = width_reduction_pct(ibp_width, alpha_width);
        eprintln!(
            "{label}: full-graph IBP {:.6} -> alpha-CROWN {:.6} ({reduction:.1}% reduction)",
            ibp_width, alpha_width,
        );
    }
    eprintln!(
        "alpha stage-local runtime: {alpha_elapsed:.1}s ({ALPHA_SMOKE_ITERATIONS} iter, {}s/stage budget)",
        ALPHA_SMOKE_STAGE_DEADLINE_SECS
    );
}

/// Alpha-CROWN compositional cosine distance: end-to-end alpha stages
/// fed through the dot/normsq suffix CROWN heads. Keep this packet focused on
/// whether the alpha compositional path itself finishes and produces sound
/// scalar bounds; the heavier CROWN-IBP comparisons already live in separate
/// tests and make this alpha lane time out. Use the same short alpha smoke
/// budget as the stage-local test; detailed bound-quality comparisons remain
/// manual `#3499` measurements.
/// Part of #3499 criterion 3.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_alpha_compositional_cosine_distance_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let config = alpha_smoke_config_for_stage();
    let alpha_start = Instant::now();
    let alpha = run_ecapa_alpha_compositional_cosine_bounds(
        graph,
        &input,
        &config,
        SPEAKER_COMPONENT_SPEC_DEADLINE_SECS,
        None,
    )
    .expect("alpha compositional cosine pipeline should succeed");
    let alpha_elapsed = alpha_start.elapsed().as_secs_f64();

    assert_finite_and_ordered(&alpha.stage_result.mfa_bounds, "alpha cosine MFA");
    assert!(
        alpha.distance_upper.is_finite(),
        "alpha distance_upper non-finite"
    );
    assert!(alpha.normsq_lower >= 0.0, "alpha normsq_lower negative");

    eprintln!(
        "alpha compositional cosine: dot=[{}, {}], normsq=[{}, {}], distance_upper={}, nonvacuous={}, runtime={alpha_elapsed:.1}s",
        alpha.dot_lower,
        alpha.dot_upper,
        alpha.normsq_lower,
        alpha.normsq_upper,
        alpha.distance_upper,
        alpha.nonvacuous,
    );
}
