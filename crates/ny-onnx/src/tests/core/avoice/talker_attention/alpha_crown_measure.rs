// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Packet C measurement: real talker-attention CROWN timing (#3588).
//!
//! After Packet A threaded GemmEngine through the pure-attention DAG
//! alpha-CROWN path, this module measures the real `talker_attention_layer0.onnx`
//! graph at the exported contract (seq_len=16, eps=1e-3).
//!
//! **Result:** Alpha-CROWN (1 iteration, SPSA) exceeds 282s on this graph
//! (3 SPSA samples × 2 perturbations = 6 CROWN evals + baseline + alpha eval).
//! The bottleneck is CROWN evaluation count, not engine threading. GPU
//! acceleration of the inner batched CROWN backward (#3597) is required
//! before alpha-CROWN is feasible at this scale.
//!
//! Reference: designs/2026-03-11-issue-3588-pure-attention-alpha-crown-runtime-path.md

use super::super::common;
use super::fixtures::{
    bounded_hidden_states_input, talker_attention_softmax_output_graph, TALKER_ATTENTION_EPSILON,
    TALKER_ATTENTION_HIDDEN_DIM, TALKER_ATTENTION_SEQ_LEN,
};
use super::*;
use ny_propagate::GraphNetwork;
use std::time::Instant;

struct BoundWidthStats {
    min: f32,
    max: f32,
    mean: f32,
}

fn bound_width_stats(bounds: &BoundedTensor) -> BoundWidthStats {
    let widths: Vec<f32> = bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| (u - l).max(0.0))
        .collect();

    let min = widths.iter().copied().fold(f32::INFINITY, f32::min);
    let max = widths.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = if widths.is_empty() {
        0.0
    } else {
        widths.iter().sum::<f32>() / widths.len() as f32
    };

    BoundWidthStats { min, max, mean }
}

/// Measure plain CROWN baseline on softmax-output graph.
fn measure_crown_baseline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> (std::time::Duration, BoundWidthStats) {
    let start = Instant::now();
    let result = graph
        .propagate_crown_with_provenance(input)
        .expect("talker attention softmax-output CROWN should succeed");
    let elapsed = start.elapsed();

    common::assert_finite_and_ordered(&result.bounds, "CROWN baseline");
    let width = bound_width_stats(&result.bounds);

    println!(
        "CROWN baseline: {:.2}s, provenance={:?}, width=[min={:.6}, max={:.6}, mean={:.6}]",
        elapsed.as_secs_f64(),
        result.provenance,
        width.min,
        width.max,
        width.mean,
    );

    (elapsed, width)
}

/// Packet C: CROWN baseline measurement on real talker-attention graph.
///
/// Confirms plain CROWN runtime at seq_len=16 and records bound widths.
/// Alpha-CROWN is documented but NOT run — see module doc for why.
///
/// Measured alpha-CROWN results (from cargo test runs):
/// - CROWN baseline: ~17.8s (1 backward pass, pre-2026-06 export, on the
///   early-fallback path — CROWN backward bailed to IBP at the RoPE
///   MulConstant nodes)
/// - Alpha-CROWN (1 iter, 3 SPSA samples): >282s (times out at 300s)
/// - Bottleneck: 8 CROWN evaluations in SPSA loop at ~35s each
/// - Conclusion: GPU acceleration (#3597) needed before alpha-CROWN is feasible
///
/// Re-measured 2026-07-19: with the MulConstant runtime-shape recovery the
/// backward pass now completes end-to-end (provenance=Crown, mean output
/// width 0.527 vs vacuous 1.0) in ~104s release under load, so the budget is
/// now 300s release-only via `release_budget_ms` (24h completes-or-hangs
/// watchdog in debug, matching the avoice smoke convention).
#[cfg_attr(not(debug_assertions), ntest::timeout(360000))]
#[test]
fn test_crown_baseline_talker_attention_seq16_3588() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (graph, softmax_name) = talker_attention_softmax_output_graph();
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let (crown_elapsed, crown_width) = measure_crown_baseline(&graph, &input);

    println!("\n=== #3588 Packet C Summary ===");
    println!("Model: talker_attention_layer0.onnx (node: {softmax_name})");
    println!(
        "Contract: seq_len={TALKER_ATTENTION_SEQ_LEN}, hidden_dim={TALKER_ATTENTION_HIDDEN_DIM}, eps={TALKER_ATTENTION_EPSILON}"
    );
    println!("CROWN: {:.2}s", crown_elapsed.as_secs_f64());
    println!("Alpha-CROWN: >282s (SPSA bottleneck, needs #3597 GPU)");
    println!(
        "width: min={:.6}, max={:.6}, mean={:.6}",
        crown_width.min, crown_width.max, crown_width.mean
    );

    // CROWN baseline budget: 300s in release (measured 103.97s under load,
    // 2026-07-19, full backward with provenance=Crown); in debug this is the
    // 24h completes-or-hangs watchdog, not a perf assertion — wall-clock
    // budgets are only meaningful in optimized builds.
    let budget_secs = common::release_budget_ms(300_000) as f64 / 1000.0;
    assert!(
        crown_elapsed.as_secs_f64() < budget_secs,
        "CROWN baseline exceeded {budget_secs:.0}s budget: {:.2}s",
        crown_elapsed.as_secs_f64()
    );
}
