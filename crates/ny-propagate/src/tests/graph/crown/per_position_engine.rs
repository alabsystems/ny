// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-position CROWN regressions for engine threading and sampled soundness.
//!
//! The engine-threading design (#3772) requires behavioral proof that
//! `propagate_crown_per_position_with_engine` and
//! `propagate_crown_within_graph_with_engine` thread the GemmEngine through
//! to actual GEMM calls. These tests use a `CountingGemmEngine` wrapper
//! to verify `gemm_calls() > 0` while preserving bound parity with the
//! bare (engine=None) variants.
//!
//! This module also covers #2557 by checking that bare per-position CROWN
//! preserves `[..., output_dim]` shape and encloses concrete outputs sampled
//! from a multi-position input tensor.

use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_test_utils::assert_bounded_tensor_close;

/// Build a simple Linear → ReLU → Linear graph for engine tests.
fn build_relu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let b1 = arr1(&[0.1_f32, -0.2, 0.3]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let b2 = arr1(&[0.0_f32, 0.1]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid Linear2")),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    graph
}

/// Evaluate the test graph at one concrete 2D point.
fn eval_relu_graph(x: &[f32; 2]) -> [f32; 2] {
    let hidden = [
        1.0 * x[0] + (-0.5) * x[1] + 0.1,
        0.5 * x[0] + 1.0 * x[1] - 0.2,
        -x[0] + 0.3 * x[1] + 0.3,
    ];
    let relu = [hidden[0].max(0.0), hidden[1].max(0.0), hidden[2].max(0.0)];
    [
        0.5 * relu[0] + (-0.3) * relu[1] + 0.8 * relu[2],
        0.2 * relu[0] + 0.6 * relu[1] + (-0.4) * relu[2] + 0.1,
    ]
}

fn assert_multi_position_sample_contained(bounds: &BoundedTensor, sample: &ArrayD<f32>, t: f32) {
    for batch_idx in 0..2 {
        for pos_idx in 0..2 {
            let point = [
                sample[[batch_idx, pos_idx, 0]],
                sample[[batch_idx, pos_idx, 1]],
            ];
            let output = eval_relu_graph(&point);
            for (out_idx, &value) in output.iter().enumerate() {
                assert!(
                    value >= bounds.lower()[[batch_idx, pos_idx, out_idx]] - 1e-5,
                    "#2557 soundness: t={t} point {:?} output[{batch_idx},{pos_idx},{out_idx}]={value:.6} < lower={:.6}",
                    point,
                    bounds.lower()[[batch_idx, pos_idx, out_idx]],
                );
                assert!(
                    value <= bounds.upper()[[batch_idx, pos_idx, out_idx]] + 1e-5,
                    "#2557 soundness: t={t} point {:?} output[{batch_idx},{pos_idx},{out_idx}]={value:.6} > upper={:.6}",
                    point,
                    bounds.upper()[[batch_idx, pos_idx, out_idx]],
                );
            }
        }
    }
}

/// #2557 regression: multi-position per-position CROWN must preserve shape
/// and contain sampled concrete outputs from each position.
#[ntest::timeout(10000)]
#[test]
fn test_crown_per_position_multi_position_soundness_2557() {
    let graph = build_relu_graph();

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![-0.6, -0.2, -0.3, -0.1, 0.0, -0.4, 0.2, 0.1],
    )
    .expect("valid lower tensor shape");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![0.4, 0.3, 0.5, 0.2, 0.6, 0.4, 0.8, 0.5],
    )
    .expect("valid upper tensor shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid input bounds");

    let bounds = graph
        .propagate_crown_per_position(&input)
        .expect("multi-position per-position CROWN should succeed");

    assert_eq!(
        bounds.shape(),
        &[2, 2, 2],
        "#2557: per-position CROWN should preserve batch dimensions and output width"
    );

    for (idx, (&lo, &hi)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(
            lo <= hi + 1e-6,
            "#2557: inverted output interval at flat index {idx}: [{lo}, {hi}]"
        );
    }

    for t in [0.0_f32, 0.25, 0.5, 1.0] {
        let sample = lower.clone() + (&upper - &lower) * t;
        assert_multi_position_sample_contained(&bounds, &sample, t);
    }
}

/// #3772 regression: `propagate_crown_per_position_with_engine` must thread
/// the GemmEngine through to CROWN backward GEMM calls.
///
/// Uses a 2D input [2, 2] to exercise the per-position loop (positions 0 and 1).
#[ntest::timeout(10000)]
#[test]
fn test_crown_per_position_with_engine_threads_gemm_3772() {
    let graph = build_relu_graph();

    // 2D input: [2 positions, 2 features]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.5, -0.3, -0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.5, 0.3, 0.3]).unwrap(),
    )
    .expect("valid 2D input bounds");

    let baseline = graph
        .propagate_crown_per_position(&input)
        .expect("baseline per-position CROWN should succeed");

    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_crown_per_position_with_engine(&input, Some(&engine))
        .expect("engine-aware per-position CROWN should succeed");

    assert_bounds_finite(&with_engine, "per-position CROWN with engine output");
    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#3772 per-position CROWN engine parity",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3772 regression: propagate_crown_per_position_with_engine must hit GemmEngine \
         (got 0 GEMM calls)"
    );
}

/// #3772 regression: `propagate_crown_within_graph_with_engine` must thread
/// the GemmEngine through to CROWN backward GEMM calls.
#[ntest::timeout(10000)]
#[test]
fn test_crown_within_graph_with_engine_threads_gemm_3772() {
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid input bounds");

    // Pin NY_DENSE_BUDGET_MB (holding the shared env lock): within-block CROWN
    // reads the budget per call, and a concurrently-running zero-budget test's
    // window would otherwise fail this test with a spurious CpuMemoryExceeded.
    let engine = CountingGemmEngine::new();
    let (baseline, with_engine) = tests::with_crown_dense_budget_mb("2048", || {
        let baseline = graph
            .propagate_crown_within_graph(&input)
            .expect("baseline within-graph CROWN should succeed");
        let with_engine = graph
            .propagate_crown_within_graph_with_engine(&input, Some(&engine))
            .expect("engine-aware within-graph CROWN should succeed");
        (baseline, with_engine)
    });

    assert_bounds_finite(&with_engine, "within-graph CROWN with engine output");
    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#3772 within-graph CROWN engine parity",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3772 regression: propagate_crown_within_graph_with_engine must hit GemmEngine \
         (got 0 GEMM calls)"
    );
}

/// #3772 regression: `propagate_crown_within_graph_with_stats_and_engine`
/// must return the same stats as the bare variant while threading the engine.
#[ntest::timeout(10000)]
#[test]
fn test_crown_within_graph_with_stats_and_engine_parity_3772() {
    let graph = build_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid input bounds");

    // Pin NY_DENSE_BUDGET_MB (holding the shared env lock): within-block CROWN
    // reads the budget per call, and a concurrently-running zero-budget test's
    // window would otherwise fail this test with a spurious CpuMemoryExceeded.
    let engine = CountingGemmEngine::new();
    let ((baseline_bounds, baseline_stats), (engine_bounds, engine_stats)) =
        tests::with_crown_dense_budget_mb("2048", || {
            let baseline = graph
                .propagate_crown_within_graph_with_stats(&input)
                .expect("baseline within-graph CROWN with stats should succeed");
            let with_engine = graph
                .propagate_crown_within_graph_with_stats_and_engine(&input, Some(&engine))
                .expect("engine-aware within-graph CROWN with stats should succeed");
            (baseline, with_engine)
        });

    assert_bounds_finite(&engine_bounds, "within-graph CROWN stats+engine output");
    assert_bounded_tensor_close(
        &engine_bounds,
        &baseline_bounds,
        1e-6,
        "#3772 within-graph CROWN stats+engine bounds parity",
    );
    assert_eq!(
        baseline_stats.len(),
        engine_stats.len(),
        "#3772: stats count should match (baseline={}, engine={})",
        baseline_stats.len(),
        engine_stats.len()
    );
    for (b_stat, e_stat) in baseline_stats.iter().zip(engine_stats.iter()) {
        assert_eq!(
            b_stat.node_name, e_stat.node_name,
            "#3772: stat node name mismatch"
        );
        assert_eq!(
            b_stat.fallback_rows, e_stat.fallback_rows,
            "#3772: stat fallback_rows mismatch for {}",
            b_stat.node_name
        );
        assert_eq!(
            b_stat.total_rows, e_stat.total_rows,
            "#3772: stat total_rows mismatch for {}",
            b_stat.node_name
        );
    }
    assert!(
        engine.gemm_calls() > 0,
        "#3772 regression: propagate_crown_within_graph_with_stats_and_engine must hit GemmEngine"
    );
}
