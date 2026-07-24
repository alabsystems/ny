// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests verifying that `collect_crown_ibp_bounds_dag_with_*_engine`
//! produces bounds equivalent to the baseline `engine=None` (faer CPU GEMM) path.
//!
//! These lock the contract from W3 commit 11ed52f (#3549, #3716):
//! three CROWN-IBP collection callers were changed from `engine=None` to
//! `engine=Some(...)` — verify bounds don't diverge.

use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_test_utils::CountingGemmEngine;

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::layer_norm::LayerNormLayer;
use crate::*;

/// Build a Linear -> ReLU -> Linear graph for CROWN-IBP engine parity tests.
pub(super) fn build_two_linear_relu_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let l1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5, -0.3], [-0.5, 1.0, 0.7], [0.3, -0.2, 1.0]]),
        Some(arr1(&[0.1_f32, -0.1, 0.05])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));

    let l2 = LinearLayer::new(
        arr2(&[[2.0_f32, -1.0, 0.5], [1.0, 2.0, -0.5]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
    graph.add_node(GraphNode::new("l2", Layer::Linear(l2), vec!["relu".into()]));
    graph.set_output("l2");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// Build a Conv1d -> ReLU graph matching the engine-threaded ECAPA Stage-A path.
fn build_conv1d_relu_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let conv_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.2, 0.4, 0.1]).unwrap();
    let conv = Conv1dLayer::with_input_length(conv_kernel, Some(arr1(&[0.15_f32, -0.05])), 1, 1, 6)
        .unwrap();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv1d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-0.5, -0.25, 0.0, -0.1, -0.2, -0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0.75, 0.5, 0.4, 0.6, 0.8, 0.7]).unwrap(),
    )
    .unwrap();
    (graph, input)
}

/// Build a 2-block FFN transformer graph for deeper CROWN-IBP engine parity tests.
fn build_two_block_ffn_graph(hidden: usize, expansion: usize) -> GraphNetwork {
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();

    let mut graph = GraphNetwork::new();

    for block_idx in 0..2 {
        let prefix = format!("layer{}", block_idx);
        let block_input_name = if block_idx == 0 {
            NETWORK_INPUT.to_string()
        } else {
            format!("layer{}_add", block_idx - 1)
        };

        let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_norm", prefix),
            Layer::LayerNorm(ln),
            vec![block_input_name.clone()],
        ));

        let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
            let phase = (i * 17 + j * 31 + block_idx * 97) as f32;
            scale1 * phase.sin() * 0.15
        });
        let linear_up = LinearLayer::new(weight_up, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_up", prefix),
            Layer::Linear(linear_up),
            vec![format!("{}_norm", prefix)],
        ));

        let gelu = GELULayer::default();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_act", prefix),
            Layer::GELU(gelu),
            vec![format!("{}_ffn_up", prefix)],
        ));

        let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
            let phase = (i * 23 + j * 37 + block_idx * 71) as f32;
            scale2 * phase.cos() * 0.15
        });
        let linear_down = LinearLayer::new(weight_down, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_down", prefix),
            Layer::Linear(linear_down),
            vec![format!("{}_ffn_act", prefix)],
        ));

        let add = AddLayer;
        graph.add_node(GraphNode::new(
            format!("{}_add", prefix),
            Layer::Add(add),
            vec![block_input_name, format!("{}_ffn_down", prefix)],
        ));
    }

    graph.set_output("layer1_add");
    graph
}

/// Assert per-node CROWN-IBP bounds are approximately equal between two runs.
///
/// Tolerance accounts for faer (SIMD/tiled) vs NaiveCpuGemmEngine (triple-loop)
/// GEMM producing different floating-point accumulation order.
pub(super) fn assert_node_bounds_parity(
    baseline: &std::collections::HashMap<String, BoundedTensor>,
    with_engine: &std::collections::HashMap<String, BoundedTensor>,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        baseline.len(),
        with_engine.len(),
        "{label}: node count mismatch (baseline={}, engine={})",
        baseline.len(),
        with_engine.len()
    );
    for (name, baseline_bt) in baseline {
        let engine_bt = with_engine
            .get(name)
            .unwrap_or_else(|| panic!("{label}: engine result missing node '{name}'"));
        assert_eq!(
            baseline_bt.shape(),
            engine_bt.shape(),
            "{label}: shape mismatch at node '{name}'"
        );
        assert!(
            !engine_bt
                .lower()
                .iter()
                .any(|v| v.is_nan() || v.is_infinite()),
            "{label}: engine lower bounds contain NaN/Inf at node '{name}'"
        );
        assert!(
            !engine_bt
                .upper()
                .iter()
                .any(|v| v.is_nan() || v.is_infinite()),
            "{label}: engine upper bounds contain NaN/Inf at node '{name}'"
        );
        for (i, ((&bl, &bu), (&el, &eu))) in baseline_bt
            .lower()
            .iter()
            .zip(baseline_bt.upper().iter())
            .zip(engine_bt.lower().iter().zip(engine_bt.upper().iter()))
            .enumerate()
        {
            assert!(
                (bl - el).abs() <= tol,
                "{label}: lower bound mismatch at node '{name}' index {i}: \
                 baseline={bl}, engine={el}, diff={}",
                (bl - el).abs()
            );
            assert!(
                (bu - eu).abs() <= tol,
                "{label}: upper bound mismatch at node '{name}' index {i}: \
                 baseline={bu}, engine={eu}, diff={}",
                (bu - eu).abs()
            );
        }
    }
}

/// #3549 regression: `collect_crown_ibp_bounds_dag_with_engine` with
/// `NaiveCpuGemmEngine` must produce bounds approximately equal to `engine=None`.
///
/// The `engine=None` path uses faer GEMM (SIMD/tiled); `NaiveCpuGemmEngine` uses
/// a naive triple loop. Both compute C = A @ B but differ in float accumulation
/// order, so we assert approximate (not bit-exact) equality.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_engine_parity_simple_3549() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_two_linear_relu_graph();

        let baseline = graph.collect_crown_ibp_bounds_dag(&input).unwrap();
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_engine(&input, Some(&engine))
            .unwrap();

        assert_node_bounds_parity(&baseline, &with_engine, 1e-5, "simple Linear-ReLU-Linear");

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3549 regression: CROWN-IBP DAG collection should use GemmEngine, got 0 calls"
        );
    });
}

/// #3549 regression: `collect_crown_ibp_bounds_dag_with_deadline_and_engine`
/// (the exact variant modified by W3 in init.rs, bab_loop.rs, crown_batched.rs)
/// must produce bounds approximately equal to the deadline-only baseline.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_deadline_engine_parity_simple_3549() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_two_linear_relu_graph();

        let deadline = Some(Instant::now() + Duration::from_mins(1));
        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_deadline(&input, deadline)
            .unwrap();
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_deadline_and_engine(&input, deadline, Some(&engine))
            .unwrap();

        assert_node_bounds_parity(
            &baseline,
            &with_engine,
            1e-5,
            "simple Linear-ReLU-Linear with deadline",
        );

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3549 regression: deadline+engine DAG collection should use GemmEngine, got 0 calls"
        );
    });
}

/// #3718/#3499 regression: the engine-aware precomputed-IBP collector used by the
/// ECAPA stage-local helper must preserve bounds and provenance on Conv1d DAGs.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_precomputed_engine_parity_conv1d_3718() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_conv1d_relu_graph();

        let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
                &input,
                ibp_bounds.clone(),
                Some(Instant::now() + Duration::from_mins(1)),
            )
            .unwrap();
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine(
                &input,
                ibp_bounds,
                Some(Instant::now() + Duration::from_mins(1)),
                Some(&engine),
            )
            .unwrap();

        assert_node_bounds_parity(
            &baseline.bounds,
            &with_engine.bounds,
            1e-5,
            "Conv1d-ReLU precomputed CROWN-IBP",
        );

        for name in baseline.bounds.keys() {
            let baseline_prov = baseline.provenance_for_node(name);
            let engine_prov = with_engine.provenance_for_node(name);
            assert_eq!(
                baseline_prov, engine_prov,
                "#3718 regression: precomputed engine path changed provenance at node '{name}': \
                 baseline={baseline_prov:?}, engine={engine_prov:?}"
            );
        }

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3718 regression: precomputed Conv1d CROWN-IBP should dispatch GemmEngine, got 0 calls"
        );
    });
}

/// #3811 regression: width-threshold precomputed CROWN-IBP must still thread
/// the supplied engine instead of hardcoding the CPU-only collector path.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_precomputed_width_threshold_engine_parity_conv1d_3811() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_conv1d_relu_graph();

        let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
                &input,
                ibp_bounds.clone(),
                Some(Instant::now() + Duration::from_mins(1)),
                0.0,
            )
            .unwrap();
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine_and_width_threshold(
                &input,
                ibp_bounds,
                Some(Instant::now() + Duration::from_mins(1)),
                Some(&engine),
                0.0,
            )
            .unwrap();

        assert_node_bounds_parity(
            &baseline.bounds,
            &with_engine.bounds,
            1e-5,
            "Conv1d-ReLU precomputed width-threshold CROWN-IBP",
        );

        for name in baseline.bounds.keys() {
            let baseline_prov = baseline.provenance_for_node(name);
            let engine_prov = with_engine.provenance_for_node(name);
            assert_eq!(
                baseline_prov, engine_prov,
                "#3811 regression: width-threshold engine path changed provenance at node '{name}': \
                 baseline={baseline_prov:?}, engine={engine_prov:?}"
            );
        }

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3811 regression: width-threshold precomputed Conv1d CROWN-IBP should dispatch GemmEngine, got 0 calls"
        );
    });
}

/// #3549 regression: engine parity on a deeper graph (2-block FFN transformer).
///
/// This exercises the O(N^2) CROWN-IBP collection loop across multiple
/// Linear backward passes, matching the real VNN-COMP workload more closely.
#[ntest::timeout(60000)]
#[test]
fn test_crown_ibp_dag_engine_parity_two_block_ffn_3549() {
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

        let baseline = graph.collect_crown_ibp_bounds_dag(&input).unwrap();
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_engine(&input, Some(&engine))
            .unwrap();

        // Wider tolerance for deeper network (more GEMM accumulation differences).
        assert_node_bounds_parity(&baseline, &with_engine, 1e-4, "2-block FFN transformer");

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3549 regression: 2-block FFN CROWN-IBP DAG should use GemmEngine, got 0 calls"
        );
    });
}

/// #3549 regression: engine parity with provenance — verify that using an engine
/// does not change which nodes get CROWN-tightened vs. fall back to IBP.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_engine_preserves_provenance_3549() {
    let (graph, input) = build_two_linear_relu_graph();

    let baseline = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&engine))
        .unwrap();

    assert_eq!(
        baseline.fallback_count(),
        with_engine.fallback_count(),
        "#3549 regression: engine changed fallback count (baseline={}, engine={})",
        baseline.fallback_count(),
        with_engine.fallback_count()
    );

    // Per-node provenance must match.
    for name in baseline.bounds.keys() {
        let bp = baseline.provenance_for_node(name);
        let ep = with_engine.provenance_for_node(name);
        assert_eq!(
            bp, ep,
            "#3549 regression: engine changed provenance at node '{name}': \
             baseline={bp:?}, engine={ep:?}"
        );
    }
}
