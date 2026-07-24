// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

const SOFTMAX_CROWN_IBP_NODE_DEADLINE_SECS: u64 = 60;

fn monotonicity_stats_from_node_bounds(
    node_bounds: &HashMap<String, BoundedTensor>,
    softmax_name: &str,
    label: &str,
) -> CentroidMonotonicityStats {
    let softmax_bounds = node_bounds
        .get(softmax_name)
        .unwrap_or_else(|| panic!("{label}: missing softmax node bounds for '{softmax_name}'"));
    centroid_monotonicity_stats(softmax_bounds, label)
}

fn assert_monotonicity_stats_no_looser(
    label: &str,
    candidate: &CentroidMonotonicityStats,
    baseline: &CentroidMonotonicityStats,
) {
    assert_eq!(
        candidate.centroid_lower.len(),
        baseline.centroid_lower.len(),
        "{label}: row count mismatch: {} vs {}",
        candidate.centroid_lower.len(),
        baseline.centroid_lower.len()
    );
    assert_eq!(
        candidate.query_seq_len, baseline.query_seq_len,
        "{label}: query_seq_len mismatch: {} vs {}",
        candidate.query_seq_len, baseline.query_seq_len
    );
    assert!(
        candidate.violations <= baseline.violations,
        "{label}: violations worsened: {} > {}",
        candidate.violations,
        baseline.violations
    );
    assert!(
        candidate.max_gap <= baseline.max_gap + 1e-4,
        "{label}: max_gap worsened: {} > {}",
        candidate.max_gap,
        baseline.max_gap
    );
    assert!(
        candidate.avg_width <= baseline.avg_width + 1e-6,
        "{label}: avg_width worsened: {} > {}",
        candidate.avg_width,
        baseline.avg_width
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_real_rope_softmax_crown_ibp_named_node_metrics_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (graph, softmax_name) = talker_attention_softmax_output_graph_real_rope();
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let ibp_nodes = graph
        .collect_node_bounds(&input)
        .expect("real-RoPE softmax-node IBP collection should succeed");
    let deadline = Instant::now() + Duration::from_secs(SOFTMAX_CROWN_IBP_NODE_DEADLINE_SECS);
    let start = Instant::now();
    let crown_ibp_nodes = graph
        .collect_crown_ibp_bounds_dag_with_deadline(&input, Some(deadline))
        .expect(
            "real-RoPE softmax-node CROWN-IBP collection should succeed under the test deadline",
        );
    let elapsed = start.elapsed();

    let ibp = monotonicity_stats_from_node_bounds(
        &ibp_nodes,
        &softmax_name,
        "real_rope_softmax_node_ibp",
    );
    let crown_ibp = monotonicity_stats_from_node_bounds(
        &crown_ibp_nodes,
        &softmax_name,
        "real_rope_softmax_node_crown_ibp",
    );
    let materially_tighter = crown_ibp.violations < ibp.violations
        || crown_ibp.max_gap < ibp.max_gap - 1e-4
        || crown_ibp.avg_width < ibp.avg_width - 1e-6;

    println!(
        "real_rope_softmax_crown_ibp: eps={:.0e} node={} deadline={}s elapsed={:.1}s ibp=(violations={}, max_gap={:.6}, avg_width={:.6}) crown_ibp=(violations={}, max_gap={:.6}, avg_width={:.6}) delta=(violations={}, max_gap={:.6}, avg_width={:.6}) materially_tighter={materially_tighter}",
        TALKER_ATTENTION_EPSILON,
        softmax_name,
        SOFTMAX_CROWN_IBP_NODE_DEADLINE_SECS,
        elapsed.as_secs_f64(),
        ibp.violations,
        ibp.max_gap,
        ibp.avg_width,
        crown_ibp.violations,
        crown_ibp.max_gap,
        crown_ibp.avg_width,
        crown_ibp.violations as isize - ibp.violations as isize,
        crown_ibp.max_gap - ibp.max_gap,
        crown_ibp.avg_width - ibp.avg_width,
    );

    assert_monotonicity_stats_no_looser("real_rope_softmax_crown_ibp", &crown_ibp, &ibp);
}
