// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: Intermediate-node tightening via CROWN-IBP (#3497)
//!
//! Direct backward CROWN on the softmax-output graph does not expand the
//! certified radius beyond the IBP baseline (measured in `crown_boundary.rs`).
//! This module bypasses the `should_use_crown_ibp_intermediates` gate by
//! calling `collect_crown_ibp_bounds_dag_with_deadline` directly, which runs
//! CROWN backward from each intermediate node in topological order.
//! A deadline is used because the O(N²) cost of full CROWN-IBP on the
//! transformer graph exceeds practical test budgets (~300s+).
//!
//! Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md §Phase 3

use super::centroid::{
    assert_centroid_bounds_no_looser, centroid_monotonicity_stats, CentroidMonotonicityStats,
};
use super::fixtures::{
    bounded_hidden_states_input, configure_sound_softmax_modes,
    first_talker_attention_softmax_node, load_talker_attention_with_real_rope_seq_len,
    TALKER_ATTENTION_EPSILON, TALKER_ATTENTION_SEQ_LEN,
};
use super::*;
use std::time::{Duration, Instant};

/// Deadline for each CROWN-IBP collection pass. The full transformer graph
/// has O(N²) cost per node, so we cap each call at 60s and accept partial
/// tightening (remaining nodes fall back to IBP, which is sound).
const CROWN_IBP_DEADLINE_SECS: u64 = 60;

fn full_talker_attention_graph_real_rope() -> GraphNetwork {
    let model = load_talker_attention_with_real_rope_seq_len(TALKER_ATTENTION_SEQ_LEN);
    configure_sound_softmax_modes(
        model
            .to_graph_network()
            .expect("talker attention with real RoPE should convert to GraphNetwork"),
    )
}

fn ibp_softmax_stats(graph: &GraphNetwork, eps: f32) -> CentroidMonotonicityStats {
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, eps);
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP node bounds should succeed on full talker attention graph");
    let softmax_name = first_talker_attention_softmax_node(graph);
    let ibp_softmax = ibp_bounds
        .get(&softmax_name)
        .unwrap_or_else(|| panic!("IBP bounds should contain softmax node '{softmax_name}'"));
    centroid_monotonicity_stats(ibp_softmax, "IBP baseline")
}

struct CrownIbpResult {
    softmax_name: String,
    stats: CentroidMonotonicityStats,
}

fn run_crown_ibp_tightening(
    graph: &GraphNetwork,
    eps: f32,
    label: &str,
) -> Result<CrownIbpResult, String> {
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, eps);
    let deadline = Instant::now() + Duration::from_secs(CROWN_IBP_DEADLINE_SECS);
    let start = Instant::now();

    let bounds = graph
        .collect_crown_ibp_bounds_dag_with_deadline(&input, Some(deadline))
        .map_err(|e| format!("{label}: collect_crown_ibp_bounds_dag failed: {e}"))?;

    let elapsed = start.elapsed();
    let softmax_name = first_talker_attention_softmax_node(graph);
    let softmax_bounds = match bounds.get(&softmax_name) {
        Some(b) => b,
        None => {
            return Err(format!(
                "{label}: softmax node '{softmax_name}' not in CROWN-IBP bounds"
            ));
        }
    };

    let stats = centroid_monotonicity_stats(softmax_bounds, &format!("{label} CROWN-IBP"));
    println!(
        "{label}: eps={eps:.0e} elapsed={:.1}s nodes_tightened={}",
        elapsed.as_secs_f64(),
        bounds.len()
    );

    Ok(CrownIbpResult {
        softmax_name,
        stats,
    })
}

/// Sweep epsilons and optionally binary-search for the critical epsilon where
/// CROWN-IBP monotonicity breaks. Returns (last_passing, first_failing).
fn crown_ibp_critical_epsilon(
    graph: &GraphNetwork,
    epsilons: &[f32],
    bisect_iterations: usize,
) -> (f32, Option<f32>) {
    let mut last_passing = 0.0f32;
    let mut first_failing = None;

    for &eps in epsilons {
        let tightened = match run_crown_ibp_tightening(graph, eps, "crown_ibp_sweep") {
            Ok(t) => t,
            Err(e) => {
                println!("crown_ibp_sweep: eps={eps:.0e} failed: {e}");
                if first_failing.is_none() {
                    first_failing = Some(eps);
                }
                break;
            }
        };
        println!(
            "crown_ibp_sweep: eps={eps:.0e} violations={} max_gap={:.6}",
            tightened.stats.violations, tightened.stats.max_gap
        );
        if tightened.stats.violations == 0 {
            last_passing = eps;
        } else if first_failing.is_none() {
            first_failing = Some(eps);
            break;
        }
    }

    // Binary-search refinement is useful for manual measurement, but the
    // regression gate only needs the coarse sweep result to prove the
    // epsilon=1e-3 baseline. Skipping the refinement keeps the test under the
    // shared cargo timeout budget.
    if bisect_iterations > 0 {
        if let (lo_start, Some(hi_start)) = (last_passing, first_failing) {
            if lo_start > 0.0 {
                let (lo, hi) = crown_ibp_bisect(graph, lo_start, hi_start, bisect_iterations);
                println!(
                    "crown_ibp_sweep: critical_epsilon_range=[{lo:.6}, {hi:.6}]\n  \
                     Compare: IBP real-RoPE critical_epsilon ≈ 2.535e-6 (2026-06-03 export)"
                );
            }
        }
    }

    (last_passing, first_failing)
}

fn crown_ibp_bisect(
    graph: &GraphNetwork,
    mut lo: f32,
    mut hi: f32,
    iterations: usize,
) -> (f32, f32) {
    for i in 0..iterations {
        let mid = f32::midpoint(lo, hi);
        let tightened = match run_crown_ibp_tightening(graph, mid, "crown_ibp_bisect") {
            Ok(t) => t,
            Err(e) => {
                println!("crown_ibp_bisect: eps={mid:.6} iter={i} error: {e}");
                hi = mid;
                continue;
            }
        };
        if tightened.stats.violations == 0 {
            lo = mid;
        } else {
            hi = mid;
        }
        println!(
            "crown_ibp_bisect: eps={mid:.6} violations={} max_gap={:.6} → [{lo:.6}, {hi:.6}]",
            tightened.stats.violations, tightened.stats.max_gap
        );
    }
    (lo, hi)
}

/// Phase 3 gate: does CROWN-IBP tightening improve softmax centroid bounds
/// over pure IBP on the full talker attention graph?
///
/// Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md §Phase 3
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_crown_ibp_tightening_softmax_centroids_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let graph = full_talker_attention_graph_real_rope();
    let eps = TALKER_ATTENTION_EPSILON;

    let ibp = ibp_softmax_stats(&graph, eps);
    println!(
        "IBP baseline: eps={eps:.0e} violations={} max_gap={:.6} avg_width={:.6}",
        ibp.violations, ibp.max_gap, ibp.avg_width
    );

    let tightened = run_crown_ibp_tightening(&graph, eps, "crown_ibp_tightening")
        .expect("CROWN-IBP tightening should succeed on full talker attention graph");

    println!(
        "CROWN-IBP: eps={eps:.0e} violations={} max_gap={:.6} avg_width={:.6}",
        tightened.stats.violations, tightened.stats.max_gap, tightened.stats.avg_width
    );

    let improvement_pct = if ibp.avg_width > 0.0 {
        100.0 * (ibp.avg_width - tightened.stats.avg_width) / ibp.avg_width
    } else {
        0.0
    };
    println!(
        "crown_ibp_tightening: width_delta={:.2}% node={}",
        improvement_pct, tightened.softmax_name
    );

    // Soundness: CROWN-IBP bounds must not be looser than IBP.
    assert_eq!(tightened.stats.query_seq_len, ibp.query_seq_len);
    assert_centroid_bounds_no_looser(
        "CROWN-IBP vs IBP centroids",
        &tightened.stats.centroid_lower,
        &tightened.stats.centroid_upper,
        &ibp.centroid_lower,
        &ibp.centroid_upper,
        1e-4,
    );

    if tightened.stats.avg_width < ibp.avg_width - 1e-6 {
        println!("crown_ibp_tightening: IMPROVEMENT — width reduced by {improvement_pct:.2}%");
    } else {
        println!(
            "crown_ibp_tightening: NO IMPROVEMENT — IBP baseline remains the published result."
        );
    }
}

/// Phase 3 epsilon sweep: does CROWN-IBP extend the critical epsilon
/// beyond the IBP baseline?
///
/// Sweep re-anchored 2026-07-19 for the 2026-06-03 fixture re-export: the
/// current `talker_attention_layer0.onnx` carries Qwen3 q/k RmsNorm whose
/// variance floor divides ~eps-scale activations by an ~1e-3 rms, so at
/// eps=1e-3 the QK^T logit IBP spans thousands and the original gate
/// ("prove at 1e-3", IBP baseline ~1.49e-3 on the pre-June export) is out of
/// reach for any relaxation. Measured real-RoPE IBP critical epsilon on this
/// export: 2.535e-6. The sweep now brackets that value; the regression gate
/// is that CROWN-IBP certifies at least the IBP-certified 2e-6 point
/// (CROWN-IBP must not be looser than IBP), and the log reports whether it
/// extends beyond 2.535e-6.
///
/// Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md §Phase 3
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_crown_ibp_tightening_epsilon_sweep_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let graph = full_talker_attention_graph_real_rope();
    let epsilons: Vec<f32> = vec![2e-6, 3e-6, 5e-6];

    let (last_passing, first_failing) = crown_ibp_critical_epsilon(&graph, &epsilons, 0);
    println!(
        "crown_ibp_sweep: last_passing={last_passing:.0e} first_failing={:?}\n  \
         Compare: real-RoPE IBP critical_epsilon=2.535e-6 (re-measured 2026-07-19 on the \
         2026-06-03 export; pre-June export was ~1.49e-3)",
        first_failing.map(|e| format!("{e:.0e}"))
    );

    assert!(
        last_passing >= 2e-6,
        "CROWN-IBP should prove monotonicity at eps=2e-6 (inside the measured IBP-certified \
         region, critical 2.535e-6 on the 2026-06-03 export), last passing was \
         {last_passing:.0e}"
    );
}
