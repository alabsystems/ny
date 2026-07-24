// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Phase 6: Sequence length scaling (#3497)
//
// Phases 1-5 used seq_len=16 (the export default). The model has dynamic 'T'
// dimensions, so we can measure how the critical epsilon scales with sequence
// length. This characterizes the fundamental limit of the affine softmax
// relaxation: as softmax dimension grows, the certified radius shrinks because
// the relaxation must bound a higher-dimensional nonlinear function.
//
// The scaling exponent (critical_epsilon ∝ T^α) is paper-critical data for
// understanding the practical applicability of attention monotonicity
// verification.
//
// Reference: designs/2026-03-11-issue-3497-centroid-monotonicity-verification-path.md
// ---------------------------------------------------------------------------

// Sweep re-anchored 2026-07-19 for the 2026-06-03 fixture re-export (Qwen3
// q/k RmsNorm: variance floor divides ~eps-scale activations by an ~1e-3 rms,
// so eps>=1e-4 is vacuous). Measured real-RoPE critical epsilons on this
// export: T=4 → 7.22e-6, T=8 → 3.97e-6, T=16 → 2.535e-6, T=32 ∈ [1e-6, 3e-6).
// The pre-June export certified ~1.49e-3 at T=16.
const SEQ_LEN_SCALING_EPSILONS: &[f32] = &[1e-7, 3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 1e-4, 1e-3];

/// Build a softmax-output subgraph with real RoPE for a given sequence length.
fn softmax_output_graph_real_rope_seq_len(
    seq_len: usize,
) -> Result<(GraphNetwork, String), String> {
    let model = load_talker_attention_with_real_rope_seq_len(seq_len);
    let graph = configure_sound_softmax_modes(
        model
            .to_graph_network()
            .map_err(|e| format!("graph conversion failed at seq_len={seq_len}: {e}"))?,
    );
    let softmax_name = first_talker_attention_softmax_node(&graph);
    let mut softmax_graph = graph;
    softmax_graph.set_output(softmax_name.clone());
    Ok((softmax_graph, softmax_name))
}

fn evaluate_seq_len_epsilon(
    graph: &GraphNetwork,
    seq_len: usize,
    eps: f32,
    label: &str,
) -> Result<(usize, f32), String> {
    let input = bounded_hidden_states_input(seq_len, eps);
    let ibp = graph.propagate_ibp(&input).map_err(|e| e.to_string())?;
    let (lower, upper, query_seq_len) =
        centroid_bounds_from_softmax(&ibp, &format!("{label} T={seq_len} eps={eps:.0e}"));
    let gaps = centroid_monotonicity_gaps(&lower, &upper, query_seq_len);
    let violations = gaps.iter().filter(|&&gap| gap > 1e-4).count();
    let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok((violations, max_gap))
}

fn find_seq_len_pass_fail_window(
    graph: &GraphNetwork,
    seq_len: usize,
) -> Result<(f32, Option<f32>), String> {
    let mut last_passing = 0.0f32;
    let mut first_failing = None;

    for &eps in SEQ_LEN_SCALING_EPSILONS {
        match evaluate_seq_len_epsilon(graph, seq_len, eps, "seq_len_scaling") {
            Ok((violations, max_gap)) => {
                println!(
                    "seq_len_scaling: T={seq_len} eps={eps:.0e} violations={violations} max_gap={max_gap:.6}"
                );

                if violations == 0 {
                    last_passing = eps;
                } else if first_failing.is_none() {
                    first_failing = Some(eps);
                }
            }
            Err(error) => {
                println!("seq_len_scaling: T={seq_len} eps={eps:.0e} IBP failed: {error}");
                if first_failing.is_none() {
                    first_failing = Some(eps);
                }
                break;
            }
        }
    }

    Ok((last_passing, first_failing))
}

fn refine_critical_epsilon(
    graph: &GraphNetwork,
    seq_len: usize,
    mut lo: f32,
    mut hi: f32,
) -> Result<(f32, f32), String> {
    for _ in 0..10 {
        let mid = f32::midpoint(lo, hi);
        let (violations, _) = evaluate_seq_len_epsilon(graph, seq_len, mid, "seq_len_bisect")
            .map_err(|e| format!("IBP failed during bisect at T={seq_len} eps={mid}: {e}"))?;
        if violations == 0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo, hi))
}

/// Find the critical epsilon for IBP centroid monotonicity at a given seq_len.
/// Returns (critical_lo, critical_hi) from binary search, or None if the
/// baseline epsilon already fails.
fn ibp_critical_epsilon_at_seq_len(seq_len: usize) -> Result<(f32, f32), String> {
    let (graph, _) = softmax_output_graph_real_rope_seq_len(seq_len)?;
    let (last_passing, first_failing) = find_seq_len_pass_fail_window(&graph, seq_len)?;

    // Binary search refinement.
    let fail_eps = match first_failing {
        Some(f) => f,
        None => {
            // All passed — the critical epsilon is above our largest sweep value.
            return Ok((last_passing, last_passing));
        }
    };
    if last_passing == 0.0 {
        // Even the smallest epsilon failed.
        return Err(format!(
            "T={seq_len}: IBP monotonicity fails at all tested epsilons"
        ));
    }

    let (lo, hi) = refine_critical_epsilon(&graph, seq_len, last_passing, fail_eps)?;
    println!("seq_len_scaling: T={seq_len} critical_epsilon_range=[{lo:.6}, {hi:.6}]");

    Ok((lo, hi))
}

fn print_seq_len_scaling_summary(results: &[(usize, f32, f32)]) {
    println!("=== Sequence Length Scaling Summary ===");
    println!("{:<6} {:<14} {:<14}", "T", "eps_lo", "eps_hi");
    for &(t, lo, hi) in results {
        println!("{:<6} {:<14.6e} {:<14.6e}", t, lo, hi);
    }
}

fn print_scaling_exponent(results: &[(usize, f32, f32)]) {
    if results.len() < 2 {
        return;
    }

    let first = results[0];
    let last = results[results.len() - 1];
    if first.1 > 0.0 && last.1 > 0.0 && first.0 != last.0 {
        let alpha = (last.1 as f64 / first.1 as f64).ln() / (last.0 as f64 / first.0 as f64).ln();
        println!("seq_len_scaling: estimated alpha={alpha:.3} (critical_eps ∝ T^alpha)");
        println!(
            "seq_len_scaling: interpretation: {}",
            if alpha > -0.5 {
                "favorable — slow degradation with sequence length"
            } else if alpha > -1.0 {
                "moderate — linear-ish degradation"
            } else {
                "challenging — superlinear degradation"
            }
        );
    }
}

fn assert_seq_len_16_regression(results: &[(usize, f32, f32)]) {
    if let Some(&(_, lo, _)) = results.iter().find(|(t, _, _)| *t == 16) {
        // Measured 2026-07-19 on the 2026-06-03 export: critical epsilon
        // bracket [2.5352e-6, 2.5356e-6] (pre-June export: ~1.49e-3).
        assert!(
            (lo - 2.535e-6).abs() < 5e-7,
            "seq_len=16 critical epsilon {lo:.6e} deviates from known result ~2.535e-6 \
             (measured on the 2026-06-03 export)"
        );
    }
}

/// Phase 6: Sequence length scaling of centroid monotonicity critical epsilon.
///
/// Measures how the IBP-provable perturbation radius scales with attention
/// sequence length. The softmax affine relaxation looseness grows with the
/// number of classes (= seq_len for attention), so the critical epsilon is
/// expected to decrease with longer sequences.
///
/// Tests seq_len ∈ {4, 8, 16, 32} and reports the scaling relationship.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_centroid_monotonicity_seq_len_scaling_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let seq_lens: Vec<usize> = vec![4, 8, 16, 32];

    println!(
        "seq_len_scaling: measuring critical epsilon at T ∈ {:?}",
        seq_lens
    );
    println!(
        "seq_len_scaling: model=talker_attention_layer0.onnx rope=Qwen3-TTS(rope_theta=1000000, head_dim=128, rope_dim=64)"
    );
    println!();

    let mut results: Vec<(usize, f32, f32)> = Vec::new();

    for &seq_len in &seq_lens {
        println!("--- seq_len_scaling: T={seq_len} ---");
        match ibp_critical_epsilon_at_seq_len(seq_len) {
            Ok((lo, hi)) => {
                println!("seq_len_scaling: T={seq_len} critical_epsilon=[{lo:.6e}, {hi:.6e}]");
                results.push((seq_len, lo, hi));
            }
            Err(e) => {
                println!("seq_len_scaling: T={seq_len} FAILED: {e}");
            }
        }
        println!();
    }

    print_seq_len_scaling_summary(&results);
    print_scaling_exponent(&results);
    assert_seq_len_16_regression(&results);

    assert!(
        results.len() >= 2,
        "seq_len_scaling: need at least 2 data points, got {}",
        results.len()
    );
}
