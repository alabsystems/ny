// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::centroid::{centroid_bounds_from_softmax, centroid_monotonicity_gaps};
use super::super::fixtures::{
    bounded_hidden_states_input, talker_attention_softmax_output_graph, TALKER_ATTENTION_EPSILON,
    TALKER_ATTENTION_SEQ_LEN,
};

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_graph_crown_avoice_talker_attention_centroid_monotonicity_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (graph, _) = talker_attention_softmax_output_graph();
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let ibp_softmax = graph
        .propagate_ibp(&input)
        .expect("talker attention softmax-output IBP should succeed");
    let crown = graph
        .propagate_crown_with_provenance(&input)
        .expect("talker attention softmax-output CROWN should succeed");
    assert_eq!(
        crown.provenance,
        ny_propagate::types::BoundsProvenance::Crown,
        "talker attention centroid monotonicity should use backward CROWN, got {:?}",
        crown.provenance
    );

    let (ibp_lower, ibp_upper, query_seq_len) =
        centroid_bounds_from_softmax(&ibp_softmax, "talker attention softmax IBP");
    let (crown_lower, crown_upper, crown_query_seq_len) =
        centroid_bounds_from_softmax(&crown.bounds, "talker attention softmax CROWN");
    assert_eq!(crown_query_seq_len, query_seq_len);

    let ibp_gaps = centroid_monotonicity_gaps(&ibp_lower, &ibp_upper, query_seq_len);
    let crown_gaps = centroid_monotonicity_gaps(&crown_lower, &crown_upper, query_seq_len);
    let ibp_violations = ibp_gaps.iter().filter(|&&gap| gap > 1e-4).count();
    let crown_violations = crown_gaps.iter().filter(|&&gap| gap > 1e-4).count();
    let ibp_max_gap = ibp_gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let crown_max_gap = crown_gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "talker attention centroid monotonicity: groups={} query_seq_len={} ibp_violations={} ibp_max_gap={ibp_max_gap:.6} crown_violations={} crown_max_gap={crown_max_gap:.6}",
        ibp_lower.len() / query_seq_len,
        query_seq_len,
        ibp_violations,
        crown_violations,
    );

    for (idx, (&crown_gap, &ibp_gap)) in crown_gaps.iter().zip(&ibp_gaps).enumerate() {
        let group = idx / (query_seq_len - 1);
        let step = idx % (query_seq_len - 1);
        assert!(
            crown_gap <= ibp_gap + 1e-4,
            "talker attention centroid monotonicity gap worsened at group {group}, step {step}: crown={crown_gap}, ibp={ibp_gap}"
        );
    }
    assert!(
        crown_violations <= ibp_violations && crown_max_gap <= ibp_max_gap + 1e-4,
        "talker attention CROWN monotonicity should be no worse than IBP: crown violations/max_gap = {crown_violations}/{crown_max_gap}, ibp = {ibp_violations}/{ibp_max_gap}"
    );
}

/// Epsilon sweep: find the maximum epsilon at which IBP proves centroid
/// monotonicity. This quantifies the robustness radius — "monotonicity holds
/// for hidden_states perturbations up to epsilon=X".
///
/// Sweep re-anchored 2026-07-19 for the 2026-06-03 fixture re-export: the
/// current `talker_attention_layer0.onnx` carries Qwen3 q/k RmsNorm whose
/// variance floor divides ~eps-scale activations by an ~1e-3 rms, so at
/// eps>=1e-4 the QK^T logit IBP spans thousands and softmax bounds are
/// vacuous (max_gap=105.0 = the fully vacuous value at seq16). Measured
/// critical epsilon on this export: 2.759e-6 (bracket
/// [2.7588e-6, 2.7593e-6]); the pre-June export certified ~1.47e-3.
///
/// Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md Phase 2
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_centroid_monotonicity_epsilon_sweep_avoice_talker_attention_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (graph, _) = talker_attention_softmax_output_graph();

    // Geometric sweep bracketing the measured critical epsilon 2.76e-6,
    // with a vacuous-regime tail so a future tightening (or re-export)
    // surfaces as an extended passing prefix instead of a silent cap.
    let epsilons: Vec<f32> = vec![1e-7, 3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 1e-4, 1e-3];

    let mut last_passing_epsilon = 0.0f32;
    let mut first_failing_epsilon = None;

    for &eps in &epsilons {
        let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, eps);
        let ibp = match graph.propagate_ibp(&input) {
            Ok(bounds) => bounds,
            Err(e) => {
                println!("epsilon_sweep: eps={eps:.0e} IBP failed: {e}");
                break;
            }
        };

        let (lower, upper, query_seq_len) =
            centroid_bounds_from_softmax(&ibp, &format!("sweep eps={eps:.0e}"));
        let gaps = centroid_monotonicity_gaps(&lower, &upper, query_seq_len);
        let violations = gaps.iter().filter(|&&gap| gap > 1e-4).count();
        let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!("epsilon_sweep: eps={eps:.0e} violations={violations} max_gap={max_gap:.6}");

        if violations == 0 {
            last_passing_epsilon = eps;
        } else if first_failing_epsilon.is_none() {
            first_failing_epsilon = Some(eps);
        }
    }

    println!(
        "epsilon_sweep: last_passing={last_passing_epsilon:.0e} first_failing={:?}",
        first_failing_epsilon.map(|e| format!("{e:.0e}"))
    );

    // Regression gate: measured critical epsilon is 2.759e-6 on the
    // 2026-06-03 export; 2e-6 is the largest sweep point below it, and
    // asserting >= 2e-6 catches any IBP tightness regression while leaving
    // headroom for cross-platform f32 drift.
    assert!(
        last_passing_epsilon >= 2e-6,
        "IBP should prove monotonicity at epsilon=2e-6 (measured critical 2.759e-6 on the \
         2026-06-03 export), but last passing was {last_passing_epsilon:.0e}"
    );

    // Binary search between last_passing and first_failing to find the
    // critical epsilon with higher precision.
    if let Some(fail_eps) = first_failing_epsilon {
        let mut lo = last_passing_epsilon;
        let mut hi = fail_eps;
        for _ in 0..10 {
            let mid = f32::midpoint(lo, hi);
            let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, mid);
            let ibp = graph
                .propagate_ibp(&input)
                .expect("IBP should not fail during binary search");
            let (lower, upper, query_seq_len) =
                centroid_bounds_from_softmax(&ibp, &format!("bisect eps={mid:.6}"));
            let gaps = centroid_monotonicity_gaps(&lower, &upper, query_seq_len);
            let violations = gaps.iter().filter(|&&gap| gap > 1e-4).count();
            let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if violations == 0 {
                lo = mid;
            } else {
                hi = mid;
            }
            println!("bisect: eps={mid:.6} violations={violations} max_gap={max_gap:.6} → range=[{lo:.6}, {hi:.6}]");
        }
        println!(
            "epsilon_sweep: critical_epsilon_range=[{lo:.6}, {hi:.6}] (monotonicity provable up to eps={lo:.6})"
        );
    }
}
