// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn real_rope_gap_stats_from_bounds(ibp: &BoundedTensor, label: &str) -> (usize, f32) {
    let (lower, upper, query_seq_len) = centroid_bounds_from_softmax(ibp, label);
    let gaps = centroid_monotonicity_gaps(&lower, &upper, query_seq_len);
    let violations = gaps.iter().filter(|&&gap| gap > 1e-4).count();
    let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (violations, max_gap)
}

fn format_real_rope_sweep_status_label(eps: f32) -> String {
    format!("real_rope_sweep: eps={eps:.0e}")
}

fn format_real_rope_bisect_range_status(
    mid: f32,
    violations: usize,
    max_gap: f32,
    lo: f32,
    hi: f32,
) -> String {
    format!(
        "real_rope_bisect: eps={mid:.6} violations={violations} max_gap={max_gap:.6} → range=[{lo:.6}, {hi:.6}]"
    )
}

fn real_rope_gap_stats(
    graph: &GraphNetwork,
    eps: f32,
    bounds_label: &str,
    status_label: &str,
) -> Option<(usize, f32)> {
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, eps);
    let ibp = match graph.propagate_ibp(&input) {
        Ok(bounds) => bounds,
        Err(err) => {
            println!("{status_label} IBP failed: {err}");
            return None;
        }
    };
    Some(real_rope_gap_stats_from_bounds(&ibp, bounds_label))
}

fn print_real_rope_critical_epsilon_range(graph: &GraphNetwork, mut lo: f32, mut hi: f32) {
    for _ in 0..10 {
        let mid = f32::midpoint(lo, hi);
        let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, mid);
        let ibp = graph
            .propagate_ibp(&input)
            .expect("IBP should not fail during binary search");
        let (violations, max_gap) =
            real_rope_gap_stats_from_bounds(&ibp, &format!("real_rope_bisect eps={mid:.6}"));
        if violations == 0 {
            lo = mid;
        } else {
            hi = mid;
        }
        println!(
            "{}",
            format_real_rope_bisect_range_status(mid, violations, max_gap, lo, hi)
        );
    }
    println!(
        "real_rope_sweep: critical_epsilon_range=[{lo:.6}, {hi:.6}] (monotonicity provable up to eps={lo:.6})"
    );
}

#[test]
fn test_real_rope_sweep_log_labels_preserve_pre_split_format_4091() {
    assert_eq!(
        format_real_rope_sweep_status_label(1e-4),
        "real_rope_sweep: eps=1e-4"
    );
    assert_eq!(
        format_real_rope_bisect_range_status(0.00125, 0, 0.000031, 0.0010, 0.0015),
        "real_rope_bisect: eps=0.001250 violations=0 max_gap=0.000031 → range=[0.001000, 0.001500]"
    );
}

/// Epsilon at which real-RoPE IBP certifies centroid monotonicity on the
/// 2026-06-03 fixture re-export (re-measured 2026-07-19).
///
/// The original phase asserted at `TALKER_ATTENTION_EPSILON` (1e-3), measured
/// on a pre-June export. The current export carries Qwen3 q/k RmsNorm whose
/// variance floor divides ~eps-scale activations by an ~1e-3 rms; at eps=1e-3
/// the QK^T logit IBP spans thousands and the softmax bounds are vacuous
/// (max_gap=105.0 = the vacuous value at seq16). Measured real-RoPE critical
/// epsilon: 2.535e-6 (bracket [2.5352e-6, 2.5356e-6]); at 1e-6 the certified
/// margin is comfortable (max_gap=-0.422).
const REAL_ROPE_IBP_CERT_EPSILON: f32 = 1e-6;

/// Phase 4: centroid monotonicity with real RoPE frequency tables.
///
/// Compares IBP centroid bounds using real position-dependent cos/sin against
/// the identity-RoPE baseline. Reports whether positional encoding affects
/// the provability of monotonicity and the critical epsilon.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_centroid_monotonicity_real_rope_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let identity = {
        let (graph, _) = talker_attention_softmax_output_graph();
        let input =
            bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, REAL_ROPE_IBP_CERT_EPSILON);
        let ibp = graph.propagate_ibp(&input).expect("identity RoPE IBP");
        centroid_monotonicity_stats(&ibp, "identity RoPE IBP")
    };
    let real = {
        let (graph, _) = talker_attention_softmax_output_graph_real_rope();
        let input =
            bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, REAL_ROPE_IBP_CERT_EPSILON);
        let ibp = graph.propagate_ibp(&input).expect("real RoPE IBP");
        centroid_monotonicity_stats(&ibp, "real RoPE IBP")
    };

    println!(
        "real_rope_monotonicity: eps={:.0e}\n  \
         identity: violations={} max_gap={:.6} avg_width={:.6}\n  \
         real_rope: violations={} max_gap={:.6} avg_width={:.6}",
        REAL_ROPE_IBP_CERT_EPSILON,
        identity.violations,
        identity.max_gap,
        identity.avg_width,
        real.violations,
        real.max_gap,
        real.avg_width,
    );

    assert_eq!(real.query_seq_len, identity.query_seq_len);
    assert_eq!(real.centroid_lower.len(), identity.centroid_lower.len());
    assert_eq!(
        real.violations, 0,
        "real RoPE should prove monotonicity at eps={}: max_gap={}",
        REAL_ROPE_IBP_CERT_EPSILON, real.max_gap
    );
    assert!(real.max_gap <= 1e-4, "real RoPE max_gap={}", real.max_gap);
}

/// Phase 4: epsilon sweep with real RoPE frequency tables.
///
/// Finds the critical epsilon at which real-RoPE IBP proves monotonicity
/// and compares against the identity-RoPE critical epsilon from Phase 2.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_centroid_monotonicity_real_rope_epsilon_sweep_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let (graph, _) = talker_attention_softmax_output_graph_real_rope();

    // Sweep bracketing the measured real-RoPE critical epsilon 2.535e-6 on
    // the 2026-06-03 export (see REAL_ROPE_IBP_CERT_EPSILON), with a
    // vacuous-regime tail so a future tightening extends the passing prefix.
    let epsilons: Vec<f32> = vec![1e-7, 3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 1e-4, 1e-3];

    let mut last_passing_epsilon = 0.0f32;
    let mut first_failing_epsilon = None;

    for &eps in &epsilons {
        let bounds_label = format!("real_rope_sweep eps={eps:.0e}");
        let status_label = format_real_rope_sweep_status_label(eps);
        let Some((violations, max_gap)) =
            real_rope_gap_stats(&graph, eps, &bounds_label, &status_label)
        else {
            if first_failing_epsilon.is_none() {
                first_failing_epsilon = Some(eps);
            }
            break;
        };

        println!("real_rope_sweep: eps={eps:.0e} violations={violations} max_gap={max_gap:.6}");

        if violations == 0 {
            last_passing_epsilon = eps;
        } else if first_failing_epsilon.is_none() {
            first_failing_epsilon = Some(eps);
        }
    }

    assert!(
        last_passing_epsilon >= 2e-6,
        "real RoPE monotonicity should remain provable at epsilon=2e-6 (measured critical \
         2.535e-6 on the 2026-06-03 export): last passing was {last_passing_epsilon:.0e}"
    );

    // Binary search for critical epsilon if we found both pass and fail points.
    if let Some(fail_eps) = first_failing_epsilon {
        if last_passing_epsilon > 0.0 {
            print_real_rope_critical_epsilon_range(&graph, last_passing_epsilon, fail_eps);
        }
    }

    println!(
        "real_rope_sweep: last_passing={last_passing_epsilon:.0e} first_failing={:?}\n  \
         Compare: identity RoPE critical_epsilon=2.759e-6 (Phase 2, re-measured 2026-07-19 \
         on the 2026-06-03 export; pre-June export was 1.47e-3)",
        first_failing_epsilon.map(|e| format!("{e:.0e}"))
    );
}
